#!/usr/bin/env python3
"""보이는 tote corner를 사용해 축별 단계식 visual servo로 정렬한다.

DEBUG 버전
- 정렬/리커버리/무한 재측정 동작은 원본과 동일하다.
- 측정 window의 분포, Jacobian 계산 중간값, 명령 전후 odom을 출력한다.
- TR/TL.y로 전진, TOP angle로 Yaw, TR/TL.x로 좌우를 한 축씩 보정한다.
- 각 이동 후 다시 측정하며 전진 -> Yaw -> 좌우 우선순위를 반복한다.
- 전진은 최대 4 cm, 좌우는 최대 2 cm로 제한한다.
- 전체 데모 로그는 ``python -u ... 2>&1 | tee ...`` 형태로 저장한다.

측정 우선순위
- RIGHT가 충분히 보이면 TR + J_RIGHT 사용
- RIGHT가 없고 LEFT가 충분히 보이면 TL + J_LEFT 사용
- 첫 측정은 RIGHT를 우선하지만, 정렬 중 RIGHT가 사라지고 LEFT가 충분히
  보이면 반대 방향 recovery 없이 LEFT로 전환한 뒤 고정
- 고정된 LEFT까지 사라지거나 양쪽 corner가 모두 없을 때만 recovery 수행
- 아직 기준 corner가 없으면 로봇 전방(+X)으로 3 cm 이동 후 다시 측정
- recovery는 연속 최대 3회이며, 유효한 corner를 다시 찾으면 횟수 초기화
- 3회 이후에도 미검출이면 현재 위치에서 계속 재측정
- RealSense의 일시적인 RuntimeError는 측정 실패로 처리하고 계속 재측정
- abort_check를 전달하면 외부 취소 요청으로 무한 재측정을 중단할 수 있음

보정 이동 또는 recovery 이동 자체가 실패한 경우에는 False를 반환한다.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from control.mobile_controller import (
    OdometryMonitor,
    build_leg,
    initialize_mobile,
    move_leg,
    odom_pose,
    wait_for_odometry,
)
from control.robot_controller import move_both_arms, move_torso_and_head
from utils.tote_vision import detect_frame_feature, draw_feature, flush_camera, start_camera
from utils.ar_marker import RealSenseCamera


# 실제 grasp 성공 자세에서 다시 측정한 REFERENCE
TARGET_TL_X_PX = 81.780336
TARGET_TL_Y_PX = 145.415547
TARGET_TR_X_PX = 593.415041
TARGET_TR_Y_PX = 134.051319
TARGET_ANGLE_DEG = -1.361944

# cal3 Yaw + cal4 X/Y calibration 결과
# row    = [corner_x(px), corner_y(px), top_angle(deg)]
# column = [robot_x(m), robot_y(m), robot_yaw(deg)]
J_LEFT = np.asarray(
    [
        [+152.3978, +236.1155, +7.6313],
        [+543.8358,  -37.2530, -2.9826],
        [ +35.4551,   +2.2057, +0.6782],
    ],
    dtype=np.float64,
)

J_RIGHT = np.asarray(
    [
        [+258.0988, +555.5162, +9.6103],
        [+865.4941,  -12.4205, +3.0141],
        [ +35.3550,   +2.2541, +0.3542],
    ],
    dtype=np.float64,
)

# 새 reference에서 측정한 side별 TOP angle / robot Yaw 기울기
LEFT_ANGLE_PER_YAW_DEG = 0.786959
RIGHT_ANGLE_PER_YAW_DEG = 0.832450

# 파지 가능 허용 범위
FORWARD_TOL_M = 0.03
LATERAL_TOL_M = 0.04
YAW_TOL_DEG = 0.8

# 정렬 보정
MAX_CORRECTIONS = 3
MAX_TRANSLATION_STEP_M = 0.05
MAX_LATERAL_STEP_M = 0.03
MAX_YAW_STEP_DEG = 5.0
HARD_MAX_TRANSLATION_M = 0.35
HARD_MAX_YAW_DEG = 20.0
REQUIRED_STABLE_WINDOWS = 2

# 미검출 recovery: 고정 corner 방향으로 횡이동, 기준이 없으면 전진
RECOVERY_FORWARD_STEP_M = 0.035
RECOVERY_LATERAL_STEP_M = 0.02
MAX_RECOVERY_MOVES = 3
MEASURE_RETRY_WAIT_S = 0.5

# 모바일 trajectory
SETTLE_S = 0.7
ALIGN_LINEAR_SPEED = 0.08
ALIGN_ANGULAR_SPEED = 0.5
QUINTIC_PEAK = 1.875
MIN_LEG_TIME = 1.5

# 한 번의 측정 window만 사용한다.
MEASURE_FRAMES = 40
MIN_CORNER_FRAMES = 20
MEASURE_TIMEOUT_S = 4.0

WINDOW_NAME = "Tote Dual Align Recovery v6"


@dataclass
class CornerMeasurement:
    """현재 측정에서 선택된 LEFT 또는 RIGHT feature."""

    side: str
    x_px: float
    y_px: float
    angle_deg: float
    valid_frames: int


@dataclass
class PoseError:
    """grasp 기준 현재 local pose 오차."""

    x_m: float
    y_m: float
    yaw_deg: float


@dataclass
class RelativeCommand:
    """현재 로봇 frame 기준 one-shot SE(2) 명령."""

    x_m: float
    y_m: float
    yaw_deg: float


def wrap_angle_rad(angle: float) -> float:
    """각도를 -pi~pi 범위로 정규화한다."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def print_value_stats(name: str, values: list[float], unit: str) -> None:
    """측정값을 바꾸지 않고 분포 확인용 통계만 출력한다."""
    if not values:
        print(f"{name:<14}: no data")
        return

    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    print(
        f"{name:<14}: n={len(array):2d} | median={median:+9.3f} | "
        f"mean={np.mean(array):+9.3f} | std={np.std(array):7.3f} | "
        f"MAD={mad:7.3f} | min={np.min(array):+9.3f} | "
        f"max={np.max(array):+9.3f} {unit}"
    )


def print_window_stats(
    label: str,
    top_angles: list[float],
    left_xs: list[float],
    left_ys: list[float],
    left_angles: list[float],
    right_xs: list[float],
    right_ys: list[float],
    right_angles: list[float],
    both_frames: int,
    elapsed_s: float,
) -> None:
    """한 번의 측정 window에서 검출 안정성을 확인한다."""
    print()
    print("=" * 100)
    print(
        f"[MEASURE WINDOW: {label}] elapsed={elapsed_s:.2f}s | "
        f"TOP={len(top_angles)} | LEFT={len(left_xs)} | "
        f"RIGHT={len(right_xs)} | BOTH={both_frames}"
    )
    print_value_stats("TOP angle", top_angles, "deg")
    print_value_stats("TL.x", left_xs, "px")
    print_value_stats("TL.y", left_ys, "px")
    print_value_stats("LEFT angle", left_angles, "deg")
    print_value_stats("TR.x", right_xs, "px")
    print_value_stats("TR.y", right_ys, "px")
    print_value_stats("RIGHT angle", right_angles, "deg")
    print("=" * 100)


def target_feature(side: str) -> np.ndarray:
    """선택 corner에 맞는 목표 feature를 반환한다."""
    if side == "RIGHT":
        return np.asarray([TARGET_TR_X_PX, TARGET_TR_Y_PX, TARGET_ANGLE_DEG], dtype=np.float64)

    return np.asarray([TARGET_TL_X_PX, TARGET_TL_Y_PX, TARGET_ANGLE_DEG], dtype=np.float64)


def estimate_axis_errors(measurement: CornerMeasurement) -> PoseError:
    """각 feature와 가장 강하게 연결된 한 축의 오차를 직접 계산한다."""
    target = target_feature(measurement.side)
    jacobian = J_RIGHT if measurement.side == "RIGHT" else J_LEFT
    feature_error_x = measurement.x_px - float(target[0])
    feature_error_y = measurement.y_px - float(target[1])
    feature_error_angle = measurement.angle_deg - float(target[2])

    if measurement.side == "RIGHT":
        angle_per_yaw = RIGHT_ANGLE_PER_YAW_DEG
    else:
        angle_per_yaw = LEFT_ANGLE_PER_YAW_DEG

    # corner y는 전후(X), corner x는 좌우(Y), TOP angle은 Yaw에 사용한다.
    x_error_m = float(feature_error_y / jacobian[1, 0])
    y_error_m = float(feature_error_x / jacobian[0, 1])
    yaw_error_deg = float(feature_error_angle / angle_per_yaw)

    point_name = "TR" if measurement.side == "RIGHT" else "TL"
    print()
    print(f"[AXIS ERROR ESTIMATION: {measurement.side}]")
    print(
        f"target {point_name}=({target[0]:.3f}, {target[1]:.3f}) px | "
        f"angle={target[2]:+.4f} deg"
    )
    print(
        f"feature error: dx={feature_error_x:+.3f} px | "
        f"dy={feature_error_y:+.3f} px | dangle={feature_error_angle:+.4f} deg"
    )
    print(
        f"axis errors  : x={x_error_m * 100:+.3f} cm | "
        f"y={y_error_m * 100:+.3f} cm | yaw={yaw_error_deg:+.4f} deg"
    )

    return PoseError(
        x_m=x_error_m,
        y_m=y_error_m,
        yaw_deg=yaw_error_deg,
    )

def within_tolerance(error: PoseError) -> bool:
    """현재 자세가 파지 가능 허용 범위인지 판정한다."""
    position_ok = abs(error.x_m) <= FORWARD_TOL_M and abs(error.y_m) <= LATERAL_TOL_M
    yaw_ok = abs(error.yaw_deg) <= YAW_TOL_DEG
    return position_ok and yaw_ok


def command_is_reasonable(command: RelativeCommand) -> bool:
    """오검출 가능성이 높은 큰 명령만 차단한다."""
    translation_m = math.hypot(command.x_m, command.y_m)
    return translation_m <= HARD_MAX_TRANSLATION_M and abs(command.yaw_deg) <= HARD_MAX_YAW_DEG


def limit_command_step(command: RelativeCommand) -> tuple[RelativeCommand, float]:
    """현재 단계의 단일 축 보정량을 제한한다."""
    translation_m = math.hypot(command.x_m, command.y_m)
    scale = 1.0

    if translation_m > MAX_TRANSLATION_STEP_M:
        scale = min(scale, MAX_TRANSLATION_STEP_M / translation_m)

    if abs(command.yaw_deg) > MAX_YAW_STEP_DEG:
        scale = min(scale, MAX_YAW_STEP_DEG / abs(command.yaw_deg))

    return (
        RelativeCommand(
            x_m=command.x_m * scale,
            y_m=command.y_m * scale,
            yaw_deg=command.yaw_deg * scale,
        ),
        scale,
    )


def trajectory_duration(command: RelativeCommand) -> float:
    """linear/angular 속도 제한을 만족하는 trajectory 시간을 계산한다."""
    distance_m = math.hypot(command.x_m, command.y_m)
    yaw_rad = abs(math.radians(command.yaw_deg))
    linear_time = QUINTIC_PEAK * distance_m / ALIGN_LINEAR_SPEED if distance_m > 1e-8 else 0.0
    angular_time = QUINTIC_PEAK * yaw_rad / ALIGN_ANGULAR_SPEED if yaw_rad > 1e-8 else 0.0
    return max(linear_time, angular_time, MIN_LEG_TIME)


def move_one_shot(robot, monitor: OdometryMonitor, command: RelativeCommand) -> bool:
    """현재 자세 기준 x/y/yaw를 한 trajectory로 실행한다."""
    before = odom_pose(monitor.odom)
    duration = trajectory_duration(command)
    print()
    print(
        f"[MOVE REQUEST] x={command.x_m:+.4f} m | y={command.y_m:+.4f} m | "
        f"yaw={command.yaw_deg:+.3f} deg | duration={duration:.2f}s"
    )
    print(
        f"[ODOM BEFORE] x={before[0]:+.4f} m | y={before[1]:+.4f} m | "
        f"yaw={math.degrees(before[2]):+.3f} deg"
    )

    leg = build_leg(
        start=before,
        target=(command.x_m, command.y_m, math.radians(command.yaw_deg)),
        absolute=False,
        duration=duration,
        turn_direction="shortest",
    )
    move_ok = move_leg(robot, monitor, leg, settle=SETTLE_S)
    after = odom_pose(monitor.odom)

    dx_world = after[0] - before[0]
    dy_world = after[1] - before[1]
    cos_yaw = math.cos(before[2])
    sin_yaw = math.sin(before[2])
    actual_x = cos_yaw * dx_world + sin_yaw * dy_world
    actual_y = -sin_yaw * dx_world + cos_yaw * dy_world
    actual_yaw_deg = math.degrees(wrap_angle_rad(after[2] - before[2]))

    print(
        f"[ODOM AFTER ] x={after[0]:+.4f} m | y={after[1]:+.4f} m | "
        f"yaw={math.degrees(after[2]):+.3f} deg"
    )
    print(
        f"[ODOM DELTA ] x={actual_x:+.4f} m | y={actual_y:+.4f} m | "
        f"yaw={actual_yaw_deg:+.3f} deg | move_ok={move_ok}"
    )
    print(
        f"[MOVE ERROR ] x={actual_x - command.x_m:+.4f} m | "
        f"y={actual_y - command.y_m:+.4f} m | "
        f"yaw={actual_yaw_deg - command.yaw_deg:+.3f} deg"
    )
    return move_ok


def build_measurement(
    side: str,
    xs: list[float],
    ys: list[float],
    angles: list[float],
) -> CornerMeasurement:
    """한쪽 corner 표본의 median을 만든다."""
    return CornerMeasurement(
        side=side,
        x_px=float(np.median(xs)),
        y_px=float(np.median(ys)),
        angle_deg=float(np.median(angles)),
        valid_frames=len(xs),
    )


def measure_preferred_feature(
    pipeline,
    *,
    show: bool,
    label: str,
    required_side: str | None = None,
) -> CornerMeasurement | None:
    """첫 측정은 RIGHT 우선, 고정 RIGHT 소실 시 검출된 LEFT로 전환한다."""
    left_xs: list[float] = []
    left_ys: list[float] = []
    left_angles: list[float] = []
    right_xs: list[float] = []
    right_ys: list[float] = []
    right_angles: list[float] = []
    top_angles: list[float] = []
    both_frames = 0
    top_frames = 0
    start_time = time.monotonic()

    flush_camera(pipeline)

    while top_frames < MEASURE_FRAMES and time.monotonic() - start_time < MEASURE_TIMEOUT_S:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asarray(color_frame.get_data())
        feature = detect_frame_feature(image)

        if show:
            cv2.imshow(WINDOW_NAME, draw_feature(image, feature, label=label))
            cv2.waitKey(1)

        if feature is None:
            continue

        top_frames += 1
        angle_deg = float(feature.top_angle_deg)
        top_angles.append(angle_deg)

        if feature.left is not None and feature.right is not None:
            both_frames += 1

        if feature.left is not None:
            left_xs.append(float(feature.left.point[0]))
            left_ys.append(float(feature.left.point[1]))
            left_angles.append(angle_deg)

        if feature.right is not None:
            right_xs.append(float(feature.right.point[0]))
            right_ys.append(float(feature.right.point[1]))
            right_angles.append(angle_deg)

    elapsed_s = time.monotonic() - start_time
    print_window_stats(
        label,
        top_angles,
        left_xs,
        left_ys,
        left_angles,
        right_xs,
        right_ys,
        right_angles,
        both_frames,
        elapsed_s,
    )

    if required_side == "RIGHT":
        if len(right_xs) >= MIN_CORNER_FRAMES:
            return build_measurement("RIGHT", right_xs, right_ys, right_angles)

        if len(left_xs) >= MIN_CORNER_FRAMES:
            print(
                f"{label}: RIGHT 검출 부족 "
                f"({len(right_xs)}/{MIN_CORNER_FRAMES}), "
                f"LEFT {len(left_xs)} frames로 전환"
            )
            return build_measurement("LEFT", left_xs, left_ys, left_angles)

        print(
            f"{label}: RIGHT/LEFT 모두 미검출 "
            f"(TOP={top_frames}, LEFT={len(left_xs)}, RIGHT={len(right_xs)})"
        )
        return None

    if required_side == "LEFT":
        if len(left_xs) >= MIN_CORNER_FRAMES:
            return build_measurement("LEFT", left_xs, left_ys, left_angles)

        print(
            f"{label}: 고정된 LEFT 미검출 "
            f"(TOP={top_frames}, LEFT={len(left_xs)}, RIGHT={len(right_xs)})"
        )
        return None

    if len(right_xs) >= MIN_CORNER_FRAMES:
        return build_measurement("RIGHT", right_xs, right_ys, right_angles)

    if len(left_xs) >= MIN_CORNER_FRAMES:
        return build_measurement("LEFT", left_xs, left_ys, left_angles)

    print(
        f"{label}: corner 미검출 "
        f"(TOP={top_frames}, LEFT={len(left_xs)}, RIGHT={len(right_xs)})"
    )
    return None


def print_measurement(label: str, measurement: CornerMeasurement, error: PoseError) -> None:
    """선택된 feature와 추정 pose를 출력한다."""
    point_name = "TR" if measurement.side == "RIGHT" else "TL"
    print()
    print(f"[{label}] {measurement.side} 사용 ({measurement.valid_frames} frames)")
    print(f"{point_name:<10}: ({measurement.x_px:.3f}, {measurement.y_px:.3f}) px")
    print(f"TOP angle : {measurement.angle_deg:+.4f} deg")
    print(
        f"error     : x={error.x_m * 100:+.2f} cm | "
        f"y={error.y_m * 100:+.2f} cm | yaw={error.yaw_deg:+.3f} deg"
    )


class ToteAligner:
    """양쪽 corner 선택과 방향별 recovery를 사용하는 tote aligner."""

    def __init__(self, camera_serial: str | None = None, show: bool = False, *, camera=None):
        if camera is None and not camera_serial:
            raise ValueError("camera_serial 또는 공용 camera 중 하나가 필요합니다.")

        self.camera_serial = camera_serial
        self.show = show
        self._camera = camera
        self._pipeline = None

    @property
    def pipeline(self):
        if self._camera is not None:
            return self._camera.pipeline
        return self._pipeline

    @property
    def started(self) -> bool:
        return self.pipeline is not None

    def start(self) -> None:
        if self.started:
            return

        if self._camera is not None:
            raise RuntimeError("공용 camera pipeline이 시작되지 않았습니다.")

        self._pipeline = start_camera(self.camera_serial)

    def stop(self) -> None:
        if self._camera is None and self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

        if self.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    def _run_recovery(
        self,
        robot,
        monitor: OdometryMonitor,
        recovery_count: int,
        reason: str,
        active_side: str | None,
    ) -> bool:
        """고정 corner가 사라지면 그 corner 방향으로, 기준이 없으면 전진한다."""
        if active_side == "RIGHT":
            # RB-Y1 body frame에서 -Y가 오른쪽이다.
            recovery_command = RelativeCommand(
                x_m=0.0,
                y_m=-RECOVERY_LATERAL_STEP_M,
                yaw_deg=0.0,
            )
            move_text = f"오른쪽으로 {RECOVERY_LATERAL_STEP_M * 100:.1f} cm 이동"
        elif active_side == "LEFT":
            recovery_command = RelativeCommand(
                x_m=0.0,
                y_m=RECOVERY_LATERAL_STEP_M,
                yaw_deg=0.0,
            )
            move_text = f"왼쪽으로 {RECOVERY_LATERAL_STEP_M * 100:.1f} cm 이동"
        else:
            recovery_command = RelativeCommand(
                x_m=RECOVERY_FORWARD_STEP_M,
                y_m=0.0,
                yaw_deg=0.0,
            )
            move_text = f"전방으로 {RECOVERY_FORWARD_STEP_M * 100:.1f} cm 이동"

        print(
            f"Recovery {recovery_count}/{MAX_RECOVERY_MOVES}: {reason} -> "
            f"{move_text}"
        )
        return move_one_shot(robot, monitor, recovery_command)

    def _wait_before_retry(self, abort_check=None) -> None:
        """재측정 전 잠깐 기다리며 외부 취소 요청도 확인한다."""
        deadline = time.monotonic() + MEASURE_RETRY_WAIT_S

        while True:
            if abort_check is not None:
                abort_check()

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return

            time.sleep(min(0.1, remaining))

    def align(
        self,
        robot,
        monitor: OdometryMonitor,
        *,
        verify: bool = True,
        abort_check=None,
    ) -> bool:
        """미검출 시 방향별 recovery를 수행하며 허용범위에 들 때까지 정렬한다."""
        if not self.started:
            self.start()

        correction_count = 0
        recovery_count = 0
        stable_count = 0
        active_side = None

        print()
        print("#" * 100)
        print("TOTE ALIGN RECOVERY INFINITE DEBUG START")
        print(
            f"target LEFT=({TARGET_TL_X_PX:.3f}, {TARGET_TL_Y_PX:.3f}) px | "
            f"RIGHT=({TARGET_TR_X_PX:.3f}, {TARGET_TR_Y_PX:.3f}) px | "
            f"angle={TARGET_ANGLE_DEG:+.4f} deg"
        )
        print(
            f"measure: TOP {MEASURE_FRAMES} frames / corner minimum {MIN_CORNER_FRAMES} / "
            f"timeout {MEASURE_TIMEOUT_S:.1f}s"
        )
        print(
            f"tolerance: forward={FORWARD_TOL_M * 100:.1f} cm | "
            f"lateral={LATERAL_TOL_M * 100:.1f} cm | yaw={YAW_TOL_DEG:.2f} deg"
        )
        print(
            f"control: FORWARD -> YAW -> LATERAL | "
            f"stable windows={REQUIRED_STABLE_WINDOWS}"
        )
        print(
            f"recovery: side={RECOVERY_LATERAL_STEP_M * 100:.1f} cm / "
            f"forward={RECOVERY_FORWARD_STEP_M * 100:.1f} cm x "
            f"{MAX_RECOVERY_MOVES}회 | 이후 현재 위치 무한 재측정"
        )
        print(f"verify={verify} | abort_check={abort_check is not None}")
        print("#" * 100)

        while True:
            label = "BEFORE" if correction_count == 0 else f"AFTER {correction_count}"

            if abort_check is not None:
                abort_check()

            try:
                # 첫 측정은 RIGHT 우선이다. RIGHT가 화면 밖으로 사라지고 LEFT가
                # 충분히 보이면 같은 측정 window에서 LEFT로 전환해 되돌림을 막는다.
                # LEFT로 전환한 뒤에는 다시 RIGHT로 승격하지 않는다.
                required_side = active_side
                measurement = measure_preferred_feature(
                    self.pipeline,
                    show=self.show,
                    label=label,
                    required_side=required_side,
                )
            except RuntimeError as error:
                print(f"{label} 카메라 측정 오류: {error}")
                print("[ALIGN RETRY] CAMERA_RUNTIME_ERROR")
                self._wait_before_retry(abort_check)
                continue

            if measurement is None:
                stable_count = 0
                if recovery_count < MAX_RECOVERY_MOVES:
                    recovery_count += 1
                    if not self._run_recovery(
                        robot,
                        monitor,
                        recovery_count,
                        f"{active_side or 'corner'} 미검출",
                        active_side,
                    ):
                        print(
                            f"Tote 정렬 실패: recovery "
                            f"{recovery_count}/{MAX_RECOVERY_MOVES} 이동 실패"
                        )
                        print("[ALIGN RESULT] FAILED_RECOVERY_MOVE")
                        return False

                    continue

                print(
                    f"recovery {MAX_RECOVERY_MOVES}회 완료 후에도 "
                    f"{active_side or 'corner'} 미검출: 현재 위치에서 계속 재측정합니다."
                )
                print("[ALIGN RETRY] WAIT_NO_CORNER")
                self._wait_before_retry(abort_check)
                continue

            if active_side is None:
                active_side = measurement.side
                print(f"정렬 기준 side 고정: {active_side}")
            elif active_side == "RIGHT" and measurement.side == "LEFT":
                active_side = "LEFT"
                recovery_count = 0
                stable_count = 0
                print(
                    "RIGHT가 화면 가장자리에서 사라져 LEFT로 전환합니다. "
                    "이번 정렬 동안 LEFT를 고정합니다."
                )

            if recovery_count > 0:
                print(f"유효한 {measurement.side} 재검출: recovery count 초기화")
                recovery_count = 0

            error = estimate_axis_errors(measurement)
            print_measurement(label, measurement, error)

            if within_tolerance(error):
                stable_count += 1
                print(
                    f"[STABLE CHECK] {stable_count}/{REQUIRED_STABLE_WINDOWS}: "
                    "모든 축이 허용범위 안입니다."
                )

                if stable_count >= REQUIRED_STABLE_WINDOWS:
                    if correction_count == 0:
                        print("이미 tote grasp 정렬 범위 안입니다.")
                    else:
                        print(f"Tote 정렬 성공: {correction_count}회 보정")
                    print("[ALIGN RESULT] ALIGNED_WITHIN_TOLERANCE")
                    return True

                self._wait_before_retry(abort_check)
                continue

            stable_count = 0

            if correction_count >= MAX_CORRECTIONS:
                print(
                    f"{MAX_CORRECTIONS}회 보정 후에도 허용범위 밖입니다. "
                    # "보정 횟수를 초기화하고 계속 정렬합니다."
                )
                print("[ALIGN RETRY] CONTINUE_AFTER_MAX_CORRECTIONS")
                # correction_count = 0

            if abs(error.x_m) > FORWARD_TOL_M:
                stage = "FORWARD"
                raw_command = RelativeCommand(
                    x_m=-error.x_m,
                    y_m=0.0,
                    yaw_deg=0.0,
                )
                print(
                    f"[STAGE: FORWARD] x error={error.x_m * 100:+.2f} cm -> "
                    "전후 이동만 수행"
                )
            elif abs(error.yaw_deg) > YAW_TOL_DEG:
                stage = "YAW"
                raw_command = RelativeCommand(
                    x_m=0.0,
                    y_m=0.0,
                    yaw_deg=-error.yaw_deg * 0.45,
                )
                print(
                    f"[STAGE: YAW] yaw error={error.yaw_deg:+.3f} deg -> "
                    "회전만 수행"
                )
            else:
                stage = "LATERAL"
                lateral_command_m = float(
                    np.clip(
                        -error.y_m,
                        -MAX_LATERAL_STEP_M,
                        MAX_LATERAL_STEP_M,
                    )
                )
                raw_command = RelativeCommand(
                    x_m=0.0,
                    y_m=lateral_command_m,
                    yaw_deg=0.0,
                )
                print(
                    f"[STAGE: LATERAL] y error={error.y_m * 100:+.2f} cm -> "
                    "좌우 이동만 수행"
                )

            distance_m = math.hypot(raw_command.x_m, raw_command.y_m)
            print()
            print(
                f"Raw command: x={raw_command.x_m:+.4f} m | y={raw_command.y_m:+.4f} m | "
                f"yaw={raw_command.yaw_deg:+.3f} deg"
            )

            if not command_is_reasonable(raw_command):
                detail = (
                    f"stage={stage}, translation={distance_m:.3f} m, "
                    f"yaw={raw_command.yaw_deg:+.2f} deg"
                )
                print(
                    f"hard limit 초과: recovery 이동 없이 현재 위치에서 재측정합니다. "
                    f"({detail})"
                )
                print("[ALIGN RETRY] WAIT_HARD_LIMIT")
                self._wait_before_retry(abort_check)
                continue

            command, scale = limit_command_step(raw_command)
            correction_count += 1

            print(
                f"Limited command: x={command.x_m:+.4f} m | y={command.y_m:+.4f} m | "
                f"yaw={command.yaw_deg:+.3f} deg | scale={scale:.4f}"
            )

            if scale < 1.0:
                print(
                    f"보정 {correction_count}/{MAX_CORRECTIONS}: "
                    f"x={command.x_m:+.4f} m | y={command.y_m:+.4f} m | "
                    f"yaw={command.yaw_deg:+.3f} deg"
                )
            else:
                print(f"보정 {correction_count}/{MAX_CORRECTIONS}: 명령을 그대로 실행합니다.")

            if not move_one_shot(robot, monitor, command):
                print(f"Tote 정렬 실패: 모바일 보정 {correction_count}/{MAX_CORRECTIONS} 이동 실패")
                print("[ALIGN RESULT] FAILED_CORRECTION_MOVE")
                return False

            if not verify:
                print("Tote one-shot 정렬 완료: verify=False")
                print("[ALIGN RESULT] ONE_SHOT_NO_VERIFY")
                return True
            
            if correction_count == 6 :
                return True


def align_tote(
    robot,
    monitor: OdometryMonitor,
    *,
    camera_serial: str,
    verify: bool = True,
    show: bool = False,
    abort_check=None,
) -> bool:
    """단독 demo 호환용 D435 start → align → stop wrapper."""
    aligner = ToteAligner(camera_serial=camera_serial, show=show)

    try:
        aligner.start()
        return aligner.align(
            robot,
            monitor,
            verify=verify,
            abort_check=abort_check,
        )
    finally:
        aligner.stop()


def main() -> None:
    """초기 자세로 이동한 뒤 debug tote 정렬을 한 번 수행하고 종료한다."""
    address = "192.168.30.1:50051"
    camera_serial = "250122079439"
    camera_width = 640
    camera_height = 480
    camera_fps = 30

    initial_torso = np.deg2rad([0.0, 30.0, -50.0, 30.0, 0.0, 0.0]).tolist()
    head_down = np.deg2rad([0.0, 43.0]).tolist()
    before_right = np.deg2rad(
        [-38.23, -53.19, -21.31, -48.14, -63.73, 81.18, 2.39]
    ).tolist()
    before_left = np.deg2rad(
        [-38.23, 53.19, 21.31, -48.14, 63.73, 81.18, -2.39]
    ).tolist()

    head_move_time = 2.0
    arm_move_time = 2.0

    parser = argparse.ArgumentParser(
        description="RB-Y1 tote dual recovery infinite debug 정렬 테스트"
    )
    parser.add_argument("--address", default=address, help="로봇 주소")
    parser.add_argument("--model", choices=("a", "m"), default="m", help="RB-Y1 모델")
    parser.add_argument(
        "--camera-serial",
        dest="camera_serial",
        default=camera_serial,
        help="Tote 인식에 사용할 Head RealSense serial",
    )
    parser.add_argument("--show-tote", action="store_true", help="Tote 검출 화면 표시")
    args = parser.parse_args()

    robot = initialize_mobile(
        args.address,
        args.model,
        power=".*",
        servo=".*",
        unlimited=False,
    )
    monitor = OdometryMonitor()
    head_camera = RealSenseCamera(
        camera_width,
        camera_height,
        camera_fps,
        serial=args.camera_serial,
    )
    tote_aligner = ToteAligner(camera=head_camera, show=args.show_tote)
    state_update_started = False
    head_camera_started = False

    try:
        robot.start_state_update(monitor.on_state, rate=50)
        state_update_started = True

        if not wait_for_odometry(monitor):
            raise RuntimeError("Odometry를 받지 못했습니다.")

        print("초기 Torso / Head 및 양팔 BEFORE 자세로 동시에 이동")
        with ThreadPoolExecutor(max_workers=2) as executor:
            torso_head_future = executor.submit(
                move_torso_and_head,
                robot,
                initial_torso,
                head_down,
                minimum_time=head_move_time,
            )
            arms_future = executor.submit(
                move_both_arms,
                robot,
                before_right,
                before_left,
                minimum_time=arm_move_time,
            )
            torso_head_ok = torso_head_future.result()
            arms_ok = arms_future.result()

        if not torso_head_ok:
            raise RuntimeError("초기 Torso / Head 자세 이동 실패")
        if not arms_ok:
            raise RuntimeError("초기 양팔 BEFORE 자세 이동 실패")

        print(
            f"Head 카메라 시작: serial={args.camera_serial}, "
            f"{camera_width}x{camera_height}@{camera_fps}"
        )
        head_camera.start()
        head_camera_started = True

        print("Tote 정렬 시작")
        if not tote_aligner.align(robot, monitor, verify=True):
            raise RuntimeError("Tote 정렬 이동 실패")

        print("Tote 정렬 테스트 완료")

    except KeyboardInterrupt:
        print("\n사용자가 Tote 정렬 테스트를 중단했습니다.")

    except Exception as error:
        print(f"Tote 정렬 테스트 실패: {error}")

    finally:
        tote_aligner.stop()

        if head_camera_started:
            try:
                head_camera.stop()
            except Exception as error:
                print(f"Head 카메라 종료 실패: {error}")

        if state_update_started:
            try:
                robot.stop_state_update()
            except Exception:
                pass

        try:
            robot.disable_control_manager()
        except Exception:
            pass

        robot.disconnect()
        print("모든 연결과 제어를 정리했습니다.")


if __name__ == "__main__":
    main()