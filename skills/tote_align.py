#!/usr/bin/env python3
"""TOP rim + LEFT top corner(TL)로 tote grasp 자세에 one-shot 정렬한다.

입력 feature
- TL.x [px]
- TL.y [px]
- TOP angle [deg]

출력
- x [m]
- y [m]
- yaw [deg]

성공한 자동 X/Y/Yaw calibration의 local Jacobian을 사용하고 x/y/yaw를 한 SE(2) trajectory로 동시에 이동한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from control.mobile_controller import OdometryMonitor, build_leg, move_leg, odom_pose
from utils.tote_vision import LeftMeasurement, flush_camera, measure_left_feature, start_camera


# 정답 grasp 자세
TARGET_TL_X_PX = 94.088
TARGET_TL_Y_PX = 114.000
TARGET_ANGLE_DEG = 0.000

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
IMAGE_ANGLE_TOL_DEG = 1.0

# 정상 동작을 막지 않고 오검출로 인한 비정상 대명령만 차단한다.
MAX_TRANSLATION_COMMAND_M = 0.10
MAX_YAW_COMMAND_DEG = 10.0

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
    position_ok = abs(error.x_m) <= POSITION_TOL_M and abs(error.y_m) <= POSITION_TOL_M
    angle_ok = abs(measurement.angle_deg - TARGET_ANGLE_DEG) <= IMAGE_ANGLE_TOL_DEG

    return position_ok and angle_ok


def command_is_reasonable(command: RelativeCommand) -> bool:
    """오검출로 인한 비정상적으로 큰 명령만 차단한다."""
    translation_m = math.hypot(command.x_m, command.y_m)

    return translation_m <= MAX_TRANSLATION_COMMAND_M and abs(command.yaw_deg) <= MAX_YAW_COMMAND_DEG


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
    """D435 stream을 데모 시작부터 종료까지 유지하고, 필요할 때 fresh tote feature만 측정해 one-shot 정렬한다."""

    def __init__(self, camera_serial: str, show: bool = False):
        self.camera_serial = camera_serial
        self.show = show
        self.pipeline = None

    @property
    def started(self) -> bool:
        """D435 pipeline이 현재 실행 중인지 반환한다."""
        return self.pipeline is not None

    def start(self) -> None:
        """D435를 한 번만 시작하며, 이후 여러 cycle에서 같은 stream을 계속 재사용한다."""
        if self.started:
            return

        self.pipeline = start_camera(self.camera_serial)

    def stop(self) -> None:
        """실행 중인 D435 pipeline과 OpenCV 창을 종료한다."""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

        if self.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    def align(self, robot, monitor: OdometryMonitor, *, verify: bool = True) -> bool:
        """현재 시점의 fresh TOP + TL feature를 측정해 x/y/yaw를 one-shot으로 보정하고 필요하면 최종 자세를 검증한다."""
        if not self.started:
            self.start()

        # 이동 중 쌓인 frame은 버리고 현재 자세의 fresh frame만 사용한다.
        flush_camera(self.pipeline)

        before = measure_left_feature(
            self.pipeline,
            frame_count=MEASURE_FRAMES,
            timeout_s=MEASURE_TIMEOUT_S,
            show=self.show,
            label="BEFORE",
        )

        if before is None:
            print("Tote 정렬 실패: TOP + TL을 안정적으로 측정하지 못했습니다.")
            return False

        before_error = estimate_pose_error(before)
        print_measurement("BEFORE", before, before_error)

        if within_tolerance(before, before_error):
            print("이미 tote grasp 정렬 범위 안입니다.")
            return True

        command = pose_error_to_command(before_error)

        print()
        print(f"One-shot command: x={command.x_m:+.4f} m | y={command.y_m:+.4f} m | yaw={command.yaw_deg:+.3f} deg")

        if not command_is_reasonable(command):
            print("Tote 정렬 실패: 비정상적으로 큰 one-shot command입니다.")
            return False

        if not move_one_shot(robot, monitor, command):
            print("Tote 정렬 실패: 모바일 이동 실패")
            return False

        if not verify:
            return True

        flush_camera(self.pipeline)

        after = measure_left_feature(
            self.pipeline,
            frame_count=MEASURE_FRAMES,
            timeout_s=MEASURE_TIMEOUT_S,
            show=self.show,
            label="AFTER",
        )

        if after is None:
            print("Tote 정렬 실패: 이동 후 TOP + TL 측정 실패")
            return False

        after_error = estimate_pose_error(after)
        print_measurement("AFTER", after, after_error)

        if within_tolerance(after, after_error):
            print("Tote one-shot 정렬 성공")
            return True

        print("Tote one-shot 정렬 실패: 최종 오차가 grasp 허용 범위를 벗어났습니다.")
        return False


def align_tote(
    robot,
    monitor: OdometryMonitor,
    *,
    camera_serial: str,
    verify: bool = True,
    show: bool = False,
) -> bool:
    """기존 단독 demo와의 호환용 함수이며, 한 번 호출할 때 D435 start → align → stop을 내부에서 수행한다."""
    aligner = ToteAligner(camera_serial=camera_serial, show=show)

    try:
        aligner.start()
        return aligner.align(robot, monitor, verify=verify)
    finally:
        aligner.stop()
