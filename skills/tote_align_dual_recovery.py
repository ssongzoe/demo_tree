#!/usr/bin/env python3
"""보이는 tote corner를 사용하고, 미검출 시 전진하며 정렬한다.

측정 우선순위
- RIGHT가 충분히 보이면 TR + J_RIGHT 사용
- RIGHT가 없고 LEFT가 충분히 보이면 TL + J_LEFT 사용
- 첫 측정에서 선택한 side를 정렬이 끝날 때까지 유지
- 둘 다 없거나 검출값이 hard limit을 만들면 로봇 전방(+X)으로 3 cm 이동 후 다시 측정
- 전진 recovery는 최대 3회

같은 위치에서 재측정하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from control.mobile_controller import OdometryMonitor, build_leg, move_leg, odom_pose
from utils.tote_vision import detect_frame_feature, draw_feature, flush_camera, start_camera


# cal4의 grasp 성공 REFERENCE
TARGET_TL_X_PX = 65.970
TARGET_TL_Y_PX = 129.936
TARGET_TR_X_PX = 595.749
TARGET_TR_Y_PX = 141.543
TARGET_ANGLE_DEG = 1.2557

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

# TOP angle은 LEFT/RIGHT 공통 feature다. 양방향 Yaw 측정의 기울기를 사용한다.
TOP_ANGLE_PER_YAW_DEG = 0.678178

# 파지 가능 허용 범위
FORWARD_TOL_M = 0.02
LATERAL_TOL_M = 0.04
IMAGE_ANGLE_TOL_DEG = 1.5

# 정렬 보정
MAX_CORRECTIONS = 3
MAX_TRANSLATION_STEP_M = 0.12
MAX_YAW_STEP_DEG = 3.0
HARD_MAX_TRANSLATION_M = 0.18
HARD_MAX_YAW_DEG = 20.0

# 미검출 recovery: 동일 위치 재시도 없이 매번 전진
RECOVERY_FORWARD_STEP_M = 0.03
MAX_RECOVERY_MOVES = 3

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

WINDOW_NAME = "Tote Dual Align Recovery v5"


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


def target_feature(side: str) -> np.ndarray:
    """선택 corner에 맞는 목표 feature를 반환한다."""
    if side == "RIGHT":
        return np.asarray([TARGET_TR_X_PX, TARGET_TR_Y_PX, TARGET_ANGLE_DEG], dtype=np.float64)

    return np.asarray([TARGET_TL_X_PX, TARGET_TL_Y_PX, TARGET_ANGLE_DEG], dtype=np.float64)


def estimate_pose_error(measurement: CornerMeasurement) -> PoseError:
    """Yaw를 먼저 구해 제거한 뒤 동일한 side의 2x2 Jacobian으로 X/Y를 계산한다."""
    feature = np.asarray(
        [measurement.x_px, measurement.y_px, measurement.angle_deg],
        dtype=np.float64,
    )
    feature_error = feature - target_feature(measurement.side)
    yaw_error_deg = float(feature_error[2] / TOP_ANGLE_PER_YAW_DEG)
    jacobian = J_RIGHT if measurement.side == "RIGHT" else J_LEFT
    corner_error = feature_error[:2] - jacobian[:2, 2] * yaw_error_deg
    xy_error = np.linalg.solve(jacobian[:2, :2], corner_error)

    return PoseError(
        x_m=float(xy_error[0]),
        y_m=float(xy_error[1]),
        yaw_deg=yaw_error_deg,
    )


def pose_error_to_command(error: PoseError) -> RelativeCommand:
    """reference pose 오차를 현재 body frame 기준 inverse SE(2) 명령으로 바꾼다."""
    yaw_rad = math.radians(error.yaw_deg)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    command_x = -(cos_yaw * error.x_m + sin_yaw * error.y_m)
    command_y = +(sin_yaw * error.x_m - cos_yaw * error.y_m)

    return RelativeCommand(x_m=command_x, y_m=command_y, yaw_deg=-error.yaw_deg)


def within_tolerance(measurement: CornerMeasurement, error: PoseError) -> bool:
    """현재 자세가 파지 가능 허용 범위인지 판정한다."""
    position_ok = abs(error.x_m) <= FORWARD_TOL_M and abs(error.y_m) <= LATERAL_TOL_M
    angle_ok = abs(measurement.angle_deg - TARGET_ANGLE_DEG) <= IMAGE_ANGLE_TOL_DEG
    return position_ok and angle_ok


def command_is_reasonable(command: RelativeCommand) -> bool:
    """오검출 가능성이 높은 큰 명령만 차단한다."""
    translation_m = math.hypot(command.x_m, command.y_m)
    return translation_m <= HARD_MAX_TRANSLATION_M and abs(command.yaw_deg) <= HARD_MAX_YAW_DEG


def limit_command_step(command: RelativeCommand) -> tuple[RelativeCommand, float]:
    """x/y/yaw 비율을 유지하며 한 번의 보정량을 제한한다."""
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
    leg = build_leg(
        start=odom_pose(monitor.odom),
        target=(command.x_m, command.y_m, math.radians(command.yaw_deg)),
        absolute=False,
        duration=trajectory_duration(command),
        turn_direction="shortest",
    )
    return move_leg(robot, monitor, leg, settle=SETTLE_S)


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
    """첫 측정은 RIGHT 우선, 이후 측정은 처음 선택한 side만 사용한다."""
    left_xs: list[float] = []
    left_ys: list[float] = []
    left_angles: list[float] = []
    right_xs: list[float] = []
    right_ys: list[float] = []
    right_angles: list[float] = []
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

        if feature.left is not None:
            left_xs.append(float(feature.left.point[0]))
            left_ys.append(float(feature.left.point[1]))
            left_angles.append(angle_deg)

        if feature.right is not None:
            right_xs.append(float(feature.right.point[0]))
            right_ys.append(float(feature.right.point[1]))
            right_angles.append(angle_deg)

    if required_side == "RIGHT":
        if len(right_xs) >= MIN_CORNER_FRAMES:
            return build_measurement("RIGHT", right_xs, right_ys, right_angles)

        print(
            f"{label}: 고정된 RIGHT 미검출 "
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
    """양쪽 corner 선택과 전진 recovery를 사용하는 tote aligner."""

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

    def _run_forward_recovery(
        self,
        robot,
        monitor: OdometryMonitor,
        recovery_count: int,
        reason: str,
    ) -> bool:
        """미검출 또는 비정상 검출이면 전방으로 3 cm 이동한다."""
        print(
            f"Recovery {recovery_count}/{MAX_RECOVERY_MOVES}: {reason} -> "
            f"전방으로 {RECOVERY_FORWARD_STEP_M * 100:.1f} cm 이동"
        )
        recovery_command = RelativeCommand(
            x_m=RECOVERY_FORWARD_STEP_M,
            y_m=0.0,
            yaw_deg=0.0,
        )
        return move_one_shot(robot, monitor, recovery_command)

    def align(self, robot, monitor: OdometryMonitor, *, verify: bool = True) -> bool:
        """한 번씩 측정하며 미검출이면 +3 cm 전진, 검출되면 선택 side로 정렬한다."""
        if not self.started:
            self.start()

        correction_count = 0
        recovery_count = 0
        active_side = None

        while True:
            label = "BEFORE" if correction_count == 0 else f"AFTER {correction_count}"
            measurement = measure_preferred_feature(
                self.pipeline,
                show=self.show,
                label=label,
                required_side=active_side,
            )

            if measurement is None:
                if recovery_count >= MAX_RECOVERY_MOVES:
                    print(f"Tote 정렬 실패: 전진 recovery {MAX_RECOVERY_MOVES}회 후에도 corner 미검출")
                    return False

                recovery_count += 1
                if not self._run_forward_recovery(
                    robot,
                    monitor,
                    recovery_count,
                    f"{active_side or 'corner'} 미검출",
                ):
                    print(f"Tote 정렬 실패: recovery {recovery_count}/{MAX_RECOVERY_MOVES} 이동 실패")
                    return False

                continue

            if active_side is None:
                active_side = measurement.side
                print(f"정렬 기준 side 고정: {active_side}")

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
                    f"{MAX_CORRECTIONS}회 보정 후 수치 허용범위 밖이지만 "
                    "실험상 파지 가능한 자세로 처리합니다."
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
                if recovery_count >= MAX_RECOVERY_MOVES:
                    print(
                        f"Tote 정렬 실패: 전진 recovery {MAX_RECOVERY_MOVES}회 후에도 "
                        f"hard limit 초과 (translation={distance_m:.3f} m, "
                        f"yaw={raw_command.yaw_deg:+.2f} deg)"
                    )
                    return False

                recovery_count += 1
                reason = (
                    f"비정상 검출/hard limit "
                    f"(translation={distance_m:.3f} m, yaw={raw_command.yaw_deg:+.2f} deg)"
                )

                if not self._run_forward_recovery(
                    robot,
                    monitor,
                    recovery_count,
                    reason,
                ):
                    print(f"Tote 정렬 실패: recovery {recovery_count}/{MAX_RECOVERY_MOVES} 이동 실패")
                    return False

                continue

            command, scale = limit_command_step(raw_command)
            correction_count += 1

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
                return False

            if not verify:
                print("Tote one-shot 정렬 완료: verify=False")
                return True


def align_tote(
    robot,
    monitor: OdometryMonitor,
    *,
    camera_serial: str,
    verify: bool = True,
    show: bool = False,
) -> bool:
    """단독 demo 호환용 D435 start → align → stop wrapper."""
    aligner = ToteAligner(camera_serial=camera_serial, show=show)

    try:
        aligner.start()
        return aligner.align(robot, monitor, verify=verify)
    finally:
        aligner.stop()
