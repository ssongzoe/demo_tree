#!/usr/bin/env python3
"""tote_align의 무한 재측정 버전: 박스 인식 실패로 데모를 끝내지 않는다.

TOP rim + LEFT top corner(TL)로 tote grasp 자세에 제한된 반복 보정으로 정렬한다.
정렬 알고리즘(Jacobian, 허용 범위, soft/hard limit, 보정 횟수)은 tote_align과 같고,
인식 실패 처리만 다르다.

tote_align과의 차이
- 미검출: 2회 시도 후 실패하지 않고 같은 자리에서 성공할 때까지 재측정 (MEASURE_ATTEMPTS = 0)
- 카메라 오류: RealSense frame timeout 같은 RuntimeError를 미검출과 같게 흡수해 재측정
- hard limit 초과: 실패로 끝내지 않고 오검출로 보고 재측정 (보정 횟수는 늘리지 않음)
- align(abort_check=...): 무한 대기를 끊는 훅. WCS 취소 요청이 오면 예외를 던져 빠져나온다
- align_and_confirm(): 정렬 후 파지 직전에 같은 자세에서 한 번 더 측정해 허용 범위를 재확인하고,
  벗어나면 FINAL_CHECK_ROUNDS만큼 재정렬한다 (tote_align에는 이 단계가 없다)
- 보정 이동 실패는 tote_align과 같이 False로 끝난다

입력 feature
- TL.x [px]
- TL.y [px]
- TOP angle [deg]

출력
- x [m]
- y [m]
- yaw [deg]

성공한 자동 X/Y/Yaw calibration의 local Jacobian을 사용한다.
한 번의 명령이 크면 안전한 step으로 제한하고, 이동 후 재측정하며 최대 3회 보정한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from control.mobile_controller import OdometryMonitor, build_leg, move_leg, odom_pose
from utils.tote_vision import LeftMeasurement, flush_camera, measure_left_feature, start_camera


# 새 grasp 성공 자세의 Robust detector median
TARGET_TL_X_PX = 80.363
TARGET_TL_Y_PX = 152.913
TARGET_ANGLE_DEG = -1.190

# 자동 calibration 결과
# row    = [TL.x(px), TL.y(px), top_angle(deg)]
# column = [robot_x(m), robot_y(m), robot_yaw(deg)]
J_LEFT = np.asarray(
    [
        [-206.0282, +894.0152, +14.7188],
        [+616.1102,  -21.1683,  -3.1169],
        [  +0.0000,   +0.0000,  +0.7121],
    ],
    dtype=np.float64,
)

J_LEFT_INV = np.linalg.inv(J_LEFT)

# 최종 grasp 허용 범위
POSITION_TOL_M = 0.02
IMAGE_ANGLE_TOL_DEG = 1.5

FORWARD_TOL_M = 0.015
LATERAL_TOL_M = 0.04

# 한 번에 움직일 soft limit과 오검출을 차단할 hard limit을 분리한다.
MAX_CORRECTIONS = 2
# 인식 실패 재시도 횟수. 0이면 성공할 때까지 무한 재측정한다 (인식 실패로 데모를 끝내지 않는다).
MEASURE_ATTEMPTS = 0
# hard limit을 넘는 오검출이 나왔을 때 재측정 횟수. 0이면 무한.
MISDETECT_ATTEMPTS = 0
# 재측정 사이 대기. 대기 중에도 취소 요청에 반응하도록 잘게 쪼개 기다린다.
MEASURE_RETRY_WAIT_S = 0.5
# 파지 직전 최종 확인이 허용 범위를 벗어났을 때 정렬을 다시 수행할 최대 횟수.
FINAL_CHECK_ROUNDS = 10
# True면 끝내 범위를 벗어날 때 파지하지 않고 정렬 실패로 끝낸다. False면 경고만 남기고 파지한다.
FINAL_CHECK_REQUIRED = False
MAX_TRANSLATION_STEP_M = 0.07
MAX_YAW_STEP_DEG = 7.0
HARD_MAX_TRANSLATION_M = 0.18
HARD_MAX_YAW_DEG = 20.0

# 모바일 trajectory
SETTLE_S = 0.7
ALIGN_LINEAR_SPEED = 0.08
ALIGN_ANGULAR_SPEED = 0.5
QUINTIC_PEAK = 1.875
MIN_LEG_TIME = 1.5

MEASURE_FRAMES = 40
MEASURE_TIMEOUT_S = 10.0


@dataclass
class PoseError:
    """정답 grasp 자세 기준 현재 local pose 오차."""

    x_m: float
    y_m: float
    yaw_deg: float


@dataclass
class RelativeCommand:
    """현재 로봇 frame 기준 one-shot SE(2) 명령."""

    x_m: float
    y_m: float
    yaw_deg: float


def estimate_pose_error(measurement: LeftMeasurement) -> PoseError:
    """LEFT feature 오차를 Jacobian inverse로 local pose 오차로 변환한다."""
    feature_error = np.asarray(
        [
            measurement.tl_x_px - TARGET_TL_X_PX,
            measurement.tl_y_px - TARGET_TL_Y_PX,
            measurement.angle_deg - TARGET_ANGLE_DEG,
        ],
        dtype=np.float64,
    )

    pose_error = J_LEFT_INV @ feature_error

    return PoseError(x_m=float(pose_error[0]), y_m=float(pose_error[1]), yaw_deg=float(pose_error[2]))


def pose_error_to_command(error: PoseError) -> RelativeCommand:
    """reference frame pose 오차를 현재 body frame 기준 inverse SE(2) 명령으로 변환한다."""
    yaw_rad = math.radians(error.yaw_deg)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    command_x = -(cos_yaw * error.x_m + sin_yaw * error.y_m)
    command_y = +(sin_yaw * error.x_m - cos_yaw * error.y_m)

    return RelativeCommand(x_m=command_x, y_m=command_y, yaw_deg=-error.yaw_deg)


def within_tolerance(measurement: LeftMeasurement, error: PoseError) -> bool:
    """현재 자세가 실제 grasp 성공 허용 범위 안인지 판정한다."""
    position_ok = (
        abs(error.x_m) <= FORWARD_TOL_M
        and abs(error.y_m) <= LATERAL_TOL_M
    )
    angle_ok = abs(measurement.angle_deg - TARGET_ANGLE_DEG) <= IMAGE_ANGLE_TOL_DEG

    return position_ok and angle_ok


def command_is_reasonable(command: RelativeCommand) -> bool:
    """오검출 가능성이 높은 hard limit 초과 명령만 차단한다."""
    translation_m = math.hypot(command.x_m, command.y_m)

    return translation_m <= HARD_MAX_TRANSLATION_M and abs(command.yaw_deg) <= HARD_MAX_YAW_DEG


def limit_command_step(command: RelativeCommand) -> tuple[RelativeCommand, float]:
    """x/y/yaw 비율을 유지하면서 한 번의 보정 명령을 soft limit 안으로 줄인다."""
    translation_m = math.hypot(command.x_m, command.y_m)
    scale = 1.0

    if translation_m > MAX_TRANSLATION_STEP_M:
        scale = min(scale, MAX_TRANSLATION_STEP_M / translation_m)

    if abs(command.yaw_deg) > MAX_YAW_STEP_DEG:
        scale = min(scale, MAX_YAW_STEP_DEG / abs(command.yaw_deg))

    limited = RelativeCommand(
        x_m=command.x_m * scale,
        y_m=command.y_m * scale,
        yaw_deg=command.yaw_deg * scale,
    )

    return limited, scale


def trajectory_duration(command: RelativeCommand) -> float:
    """동시 x/y/yaw 이동에서 linear/angular 속도 제한을 모두 만족하는 시간을 선택한다."""
    distance_m = math.hypot(command.x_m, command.y_m)
    yaw_rad = abs(math.radians(command.yaw_deg))

    linear_time = QUINTIC_PEAK * distance_m / ALIGN_LINEAR_SPEED if distance_m > 1e-8 else 0.0
    angular_time = QUINTIC_PEAK * yaw_rad / ALIGN_ANGULAR_SPEED if yaw_rad > 1e-8 else 0.0

    return max(linear_time, angular_time, MIN_LEG_TIME)


def move_one_shot(robot, monitor: OdometryMonitor, command: RelativeCommand) -> bool:
    """현재 자세 기준 x/y/yaw를 한 trajectory로 동시에 실행한다."""
    leg = build_leg(
        start=odom_pose(monitor.odom),
        target=(command.x_m, command.y_m, math.radians(command.yaw_deg)),
        absolute=False,
        duration=trajectory_duration(command),
        turn_direction="shortest",
    )

    return move_leg(robot, monitor, leg, settle=SETTLE_S)


def print_measurement(label: str, measurement: LeftMeasurement, error: PoseError) -> None:
    """정렬 전/후 feature와 추정 pose를 출력한다."""
    print()
    print(f"[{label}]")
    print(f"TL        : ({measurement.tl_x_px:.3f}, {measurement.tl_y_px:.3f}) px")
    print(f"TOP angle : {measurement.angle_deg:+.4f} deg")
    print(f"error     : x={error.x_m * 100:+.2f} cm | y={error.y_m * 100:+.2f} cm | yaw={error.yaw_deg:+.3f} deg")


class ToteAligner:
    """자체 pipeline 또는 main의 공용 pipeline으로 tote를 측정하고 최대 3회 보정한다."""

    def __init__(self, camera_serial: str | None = None, show: bool = False, *, camera=None):
        if camera is None and not camera_serial:
            raise ValueError("camera_serial 또는 공용 camera 중 하나가 필요합니다.")

        self.camera_serial = camera_serial
        self.show = show
        self._camera = camera
        self._pipeline = None

    @property
    def pipeline(self):
        """공용 camera를 사용하면 현재 camera.pipeline을 동적으로 반환한다."""
        if self._camera is not None:
            return self._camera.pipeline
        return self._pipeline

    @property
    def started(self) -> bool:
        """측정에 사용할 pipeline 객체가 준비됐는지 반환한다."""
        return self.pipeline is not None

    def start(self) -> None:
        """단독 사용 시에만 D435를 시작한다. 공용 camera의 start는 main이 담당한다."""
        if self.started:
            return

        if self._camera is not None:
            raise RuntimeError(
                "공용 camera pipeline이 시작되지 않았습니다. "
                "main()에서 먼저 camera.start() 하세요."
            )

        self._pipeline = start_camera(self.camera_serial)

    def stop(self) -> None:
        """단독 사용 시에만 pipeline을 종료하며, 공용 camera는 main의 소유로 남긴다."""
        if self._camera is None and self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

        if self.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    def _wait_before_retry(self, abort_check=None) -> None:
        """재측정 전 잠깐 기다린다. 대기 중에도 취소 요청에 반응하도록 잘게 쪼갠다."""
        deadline = time.monotonic() + MEASURE_RETRY_WAIT_S
        while True:
            if abort_check is not None:
                abort_check()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(0.1, remaining))

    def _measure_once(self, label: str) -> LeftMeasurement | None:
        """한 번 측정한다. 카메라 오류는 미검출과 같게 다뤄 재측정으로 넘긴다."""
        try:
            flush_camera(self.pipeline)
            return measure_left_feature(
                self.pipeline,
                frame_count=MEASURE_FRAMES,
                timeout_s=MEASURE_TIMEOUT_S,
                show=self.show,
                label=label,
            )
        except RuntimeError as error:  # RealSense frame timeout 등 일시적 카메라 오류
            print(f"{label} 카메라 측정 오류: {error}")
            return None

    def _measure_with_retry(self, label: str, abort_check=None) -> LeftMeasurement | None:
        """현재 자세의 fresh frame을 사용하며, 검출 실패 시 움직이지 않고 다시 측정한다.

        MEASURE_ATTEMPTS가 0이면 성공할 때까지 무한 재측정하므로 None을 돌려주지 않는다.
        abort_check는 매 시도와 대기 중에 호출되며, 예외를 던져 무한 대기를 끊을 수 있다.
        """
        attempt = 0
        started = time.monotonic()

        while True:
            if abort_check is not None:
                abort_check()

            attempt += 1
            measurement = self._measure_once(label)

            if measurement is not None:
                if attempt > 1:
                    print(f"{label} TOP + TL 측정 성공: {attempt}회째 시도 "
                          f"({time.monotonic() - started:.0f}초 경과)")
                return measurement

            if MEASURE_ATTEMPTS and attempt >= MEASURE_ATTEMPTS:
                return None

            again = (f"{attempt + 1}/{MEASURE_ATTEMPTS}" if MEASURE_ATTEMPTS
                     else f"{attempt + 1}회째 (성공할 때까지 반복)")
            print(f"{label} TOP + TL 측정 실패: 카메라 재측정 {again} "
                  f"[{time.monotonic() - started:.0f}초 경과]")
            self._wait_before_retry(abort_check)

    def align(self, robot, monitor: OdometryMonitor, *, verify: bool = True,
              abort_check=None) -> bool:
        """fresh TOP + TL을 반복 측정하며 x/y/yaw를 최대 MAX_CORRECTIONS회 제한 보정한다.

        인식 실패와 hard limit을 넘는 오검출은 실패로 끝내지 않고 같은 자리에서 계속 재측정한다
        (MEASURE_ATTEMPTS / MISDETECT_ATTEMPTS가 0일 때). 보정 이동 실패만 False로 끝난다.
        abort_check는 재측정 대기를 끊는 훅이며, 던진 예외는 그대로 호출부로 전달된다.
        """
        if not self.started:
            self.start()

        correction_count = 0
        misdetect_count = 0

        while True:
            label = "BEFORE" if correction_count == 0 else f"AFTER {correction_count}"
            measurement = self._measure_with_retry(label, abort_check=abort_check)

            if measurement is None:
                print(f"Tote 정렬 실패: {label} TOP + TL을 안정적으로 측정하지 못했습니다.")
                return False

            error = estimate_pose_error(measurement)
            print_measurement(label, measurement, error)

            if within_tolerance(measurement, error):
                if correction_count == 0:
                    print("이미 tote grasp 정렬 범위 안입니다.")
                else:
                    print(f"Tote 정렬 성공: {correction_count}회 보정")
                return True

            if correction_count >= MAX_CORRECTIONS:
                print(
                    f"Tote 정렬 실패: {MAX_CORRECTIONS}회 보정 후에도 "
                    "grasp 허용 범위를 벗어났습니다."
                )
                return True

            raw_command = pose_error_to_command(error)
            distance_m = math.hypot(raw_command.x_m, raw_command.y_m)

            print()
            print(
                f"Raw command: x={raw_command.x_m:+.4f} m | y={raw_command.y_m:+.4f} m | "
                f"yaw={raw_command.yaw_deg:+.3f} deg"
            )

            if not command_is_reasonable(raw_command):
                # 오검출로 보고 보정 횟수를 늘리지 않은 채 같은 자리에서 다시 측정한다.
                misdetect_count += 1
                detail = f"translation={distance_m:.3f} m, yaw={raw_command.yaw_deg:+.2f} deg"

                if MISDETECT_ATTEMPTS and misdetect_count >= MISDETECT_ATTEMPTS:
                    print(f"Tote 정렬 실패: hard limit을 넘는 명령이 "
                          f"{misdetect_count}/{MISDETECT_ATTEMPTS}회 반복됐습니다. {detail}")
                    return False

                again = (f"{misdetect_count}/{MISDETECT_ATTEMPTS}회째" if MISDETECT_ATTEMPTS
                         else f"{misdetect_count}회째 (성공할 때까지 반복)")
                print(f"hard limit을 넘는 명령입니다: 오검출로 보고 재측정 {again}. {detail}")
                self._wait_before_retry(abort_check)
                continue

            command, scale = limit_command_step(raw_command)
            correction_number = correction_count + 1

            if scale < 1.0:
                print(
                    f"보정 {correction_number}/{MAX_CORRECTIONS}: soft limit 적용 -> "
                    f"x={command.x_m:+.4f} m | y={command.y_m:+.4f} m | yaw={command.yaw_deg:+.3f} deg"
                )
            else:
                print(f"보정 {correction_number}/{MAX_CORRECTIONS}: 명령을 그대로 실행합니다.")

            if not move_one_shot(robot, monitor, command):
                print(f"Tote 정렬 실패: 모바일 보정 {correction_number}/{MAX_CORRECTIONS} 이동 실패")
                return False

            if not verify:
                print("Tote one-shot 정렬 완료: verify=False")
                return True

            correction_count += 1
    def confirm(self, *, label: str = "CONFIRM", abort_check=None) -> bool:
        """현재 자세에서 한 번 더 측정해 grasp 허용 범위 안인지 확인한다. 로봇을 움직이지 않는다."""
        if not self.started:
            self.start()

        measurement = self._measure_with_retry(label, abort_check=abort_check)
        if measurement is None:
            print(f"파지 전 최종 확인 실패: {label} TOP + TL을 측정하지 못했습니다.")
            return False

        error = estimate_pose_error(measurement)
        print_measurement(label, measurement, error)

        if within_tolerance(measurement, error):
            print("파지 전 최종 확인: grasp 허용 범위 안입니다.")
            return True

        print("파지 전 최종 확인: grasp 허용 범위를 벗어났습니다.")
        return False

    def align_and_confirm(self, robot, monitor: OdometryMonitor, *, verify: bool = True,
                          abort_check=None) -> bool:
        """정렬한 뒤 파지 직전에 같은 자세에서 한 번 더 확인한다.

        확인이 범위를 벗어나면 FINAL_CHECK_ROUNDS회까지 정렬을 다시 수행한다.
        그래도 벗어나면 FINAL_CHECK_REQUIRED가 True면 False, False면 경고만 남기고 True.
        측정 자체가 안 되는 경우는 _measure_with_retry가 계속 재측정하므로 여기서 끝나지 않는다.
        """
        for round_index in range(1, FINAL_CHECK_ROUNDS + 2):
            if not self.align(robot, monitor, verify=verify, abort_check=abort_check):
                return False

            if self.confirm(label=f"CONFIRM {round_index}", abort_check=abort_check):
                return True

            if round_index > FINAL_CHECK_ROUNDS:
                break

            print(f"재정렬 {round_index}/{FINAL_CHECK_ROUNDS}: 최종 확인이 범위를 벗어나 정렬을 다시 수행합니다.")

        if FINAL_CHECK_REQUIRED:
            print(f"파지 전 최종 확인 실패: 재정렬 {FINAL_CHECK_ROUNDS}회 후에도 범위를 벗어나 파지하지 않습니다.")
            return False

        print(f"경고: 재정렬 {FINAL_CHECK_ROUNDS}회 후에도 최종 확인이 범위를 벗어난 상태로 파지를 진행합니다.")
        return True


def align_tote(
    robot,
    monitor: OdometryMonitor,
    *,
    camera_serial: str,
    verify: bool = True,
    show: bool = False,
) -> bool:
    """기존 단독 demo 호환용이며 D435 start → align → stop을 내부 수행한다."""
    aligner = ToteAligner(camera_serial=camera_serial, show=show)

    try:
        aligner.start()
        return aligner.align(robot, monitor, verify=verify)
    finally:
        aligner.stop()
