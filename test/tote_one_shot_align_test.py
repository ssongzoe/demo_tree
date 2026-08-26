#!/usr/bin/env python3
"""자동 X/Y/Yaw calibration 결과를 반영해 TOP rim + TL로 SE(2) 한 번 정렬하는 V2 테스트.

전제
- 카메라/브라켓 위치는 calibration 당시와 동일하다.
- detector 파라미터는 tote_corner_calibration_test.py를 그대로 사용한다.
- LEFT feature = [TL.x, TL.y, top_angle]
- local Jacobian은 grasp 성공 기준 자세 근처의 ±2 cm / ±2 deg calibration 결과를 사용한다.

실행:
python test/tote_one_shot_align_test_v2.py --serial 250122079439

dry-run:
python test/tote_one_shot_align_test_v2.py --serial 250122079439 --dry-run
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from control.mobile_controller import OdometryMonitor, build_leg, initialize_mobile, move_leg, odom_pose, wait_for_odometry
from tote_corner_calibration_test import detect_frame_feature, draw_feature, start_camera


# 로봇

ADDRESS = "192.168.30.1:50051"

SETTLE_S = 0.7
ALIGN_ANGULAR_SPEED = 0.5
ALIGN_LINEAR_SPEED = 0.08
QUINTIC_PEAK = 1.875
MIN_LEG_TIME = 1.5

CAMERA_FLUSH_FRAMES = 10
MEASURE_FRAMES = 40
MEASURE_TIMEOUT_S = 10.0

WINDOW_NAME = "Tote One Shot Align"


# 정답 grasp 자세에서 최근 3회 reference 평균
# TL.x = 94.571, 93.617, 94.076
TARGET_TL_X_PX = 94.088
TARGET_TL_Y_PX = 114.000
TARGET_ANGLE_DEG = 0.000


# LEFT local Jacobian
#
# feature_error = J_LEFT @ pose_error
#
# feature_error = [TL.x(px), TL.y(px), top_angle(deg)]
# pose_error    = [x(m), y(m), robot_yaw(deg)]
#
# x/y column: ±2 cm 자동 odom calibration 결과
# yaw column: ±2 deg 자동 odom calibration 3회 평균
# TOP angle은 translation이 아니라 yaw에만 반응한다고 두어 x/y -> angle coupling은 0으로 둔다.
J_LEFT = np.asarray(
    [
        [-206.0282, +894.0152, +14.7188],
        [+616.1102,  -21.1683,  -3.1169],
        [  +0.0000,   +0.0000,  +0.7121],
    ],
    dtype=np.float64,
)

J_LEFT_INV = np.linalg.inv(J_LEFT)


# 검증 허용 범위
POSITION_TOL_M = 0.02
IMAGE_ANGLE_TOL_DEG = 1.0

# local calibration에서 너무 멀리 벗어난 입력은 extrapolation하지 않는다.
MAX_FORWARD_COMMAND_M = 0.05
MAX_LATERAL_COMMAND_M = 0.05
MAX_YAW_COMMAND_DEG = 8.0


@dataclass
class LeftMeasurement:
    """여러 frame에서 얻은 LEFT feature median."""

    tl_x_px: float
    tl_y_px: float
    angle_deg: float
    tl_x_std_px: float
    tl_y_std_px: float
    angle_std_deg: float
    valid_frames: int


@dataclass
class PoseError:
    """정답 grasp 자세 기준 현재 로봇의 local pose 오차."""

    x_m: float
    y_m: float
    yaw_deg: float


@dataclass
class RelativeCommand:
    """현재 로봇 자세 기준으로 정답 자세까지 가기 위한 상대 SE(2) 명령."""

    x_m: float
    y_m: float
    yaw_deg: float


def flush_camera(pipeline: rs.pipeline) -> None:
    """로봇 이동 중 쌓인 이전 frame을 버린다."""
    for _ in range(CAMERA_FLUSH_FRAMES):
        pipeline.wait_for_frames()


def measure_left(pipeline: rs.pipeline, label: str) -> LeftMeasurement | None:
    """TL + TOP이 동시에 검출된 frame만 모아 median feature를 반환한다."""
    tl_xs: list[float] = []
    tl_ys: list[float] = []
    angles: list[float] = []

    start_time = time.monotonic()

    while len(tl_xs) < MEASURE_FRAMES and time.monotonic() - start_time < MEASURE_TIMEOUT_S:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asarray(color_frame.get_data())
        feature = detect_frame_feature(image)
        output = draw_feature(image, feature)

        cv2.putText(output, label, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, output)
        cv2.waitKey(1)

        if feature is None or feature.left is None:
            continue

        tl_xs.append(float(feature.left.point[0]))
        tl_ys.append(float(feature.left.point[1]))
        angles.append(float(feature.top_angle_deg))

    if len(tl_xs) < MEASURE_FRAMES // 2:
        print(f"{label}: LEFT feature 측정 실패 ({len(tl_xs)}/{MEASURE_FRAMES})")
        return None

    measurement = LeftMeasurement(
        tl_x_px=float(np.median(tl_xs)),
        tl_y_px=float(np.median(tl_ys)),
        angle_deg=float(np.median(angles)),
        tl_x_std_px=float(np.std(tl_xs)),
        tl_y_std_px=float(np.std(tl_ys)),
        angle_std_deg=float(np.std(angles)),
        valid_frames=len(tl_xs),
    )

    print()
    print(f"[{label}]")
    print(f"TL            : ({measurement.tl_x_px:.3f}, {measurement.tl_y_px:.3f}) px")
    print(f"TOP angle     : {measurement.angle_deg:+.4f} deg")
    print(
        f"std           : TL.x={measurement.tl_x_std_px:.3f} px | TL.y={measurement.tl_y_std_px:.3f} px | "
        f"angle={measurement.angle_std_deg:.4f} deg | frames={measurement.valid_frames}"
    )

    return measurement


def estimate_pose_error(measurement: LeftMeasurement) -> PoseError:
    """현재 LEFT feature와 target의 차이를 local Jacobian inverse로 pose 오차로 변환한다."""
    feature_error = np.asarray(
        [
            measurement.tl_x_px - TARGET_TL_X_PX,
            measurement.tl_y_px - TARGET_TL_Y_PX,
            measurement.angle_deg - TARGET_ANGLE_DEG,
        ],
        dtype=np.float64,
    )

    pose_error = J_LEFT_INV @ feature_error

    return PoseError(
        x_m=float(pose_error[0]),
        y_m=float(pose_error[1]),
        yaw_deg=float(pose_error[2]),
    )


def pose_error_to_command(error: PoseError) -> RelativeCommand:
    """reference frame의 pose 오차를 현재 로봇 frame 기준 inverse SE(2) 명령으로 변환한다."""
    yaw_rad = math.radians(error.yaw_deg)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    # T_ref_current = [R(theta), t]라 두면 정답으로 가는 상대 변환은 T_current_ref = inverse(T_ref_current).
    command_x = -(cos_yaw * error.x_m + sin_yaw * error.y_m)
    command_y = +(sin_yaw * error.x_m - cos_yaw * error.y_m)

    return RelativeCommand(
        x_m=command_x,
        y_m=command_y,
        yaw_deg=-error.yaw_deg,
    )


def print_error(label: str, error: PoseError) -> None:
    """추정 pose 오차를 보기 좋게 출력한다."""
    print(f"{label} x error   : {error.x_m * 100:+.2f} cm")
    print(f"{label} y error   : {error.y_m * 100:+.2f} cm")
    print(f"{label} yaw error : {error.yaw_deg:+.3f} deg")


def print_command(command: RelativeCommand) -> None:
    """SE(2) one-shot command를 출력한다."""
    print()
    print("ONE-SHOT COMMAND")
    print(f"x    : {command.x_m:+.4f} m")
    print(f"y    : {command.y_m:+.4f} m")
    print(f"yaw  : {command.yaw_deg:+.3f} deg")


def command_is_safe(command: RelativeCommand) -> bool:
    """local calibration 범위를 지나치게 extrapolation하는 큰 명령은 실행하지 않는다."""
    if abs(command.x_m) > MAX_FORWARD_COMMAND_M:
        print(f"x command가 너무 큽니다: {command.x_m:+.4f} m > ±{MAX_FORWARD_COMMAND_M:.3f} m")
        return False

    if abs(command.y_m) > MAX_LATERAL_COMMAND_M:
        print(f"y command가 너무 큽니다: {command.y_m:+.4f} m > ±{MAX_LATERAL_COMMAND_M:.3f} m")
        return False

    if abs(command.yaw_deg) > MAX_YAW_COMMAND_DEG:
        print(f"yaw command가 너무 큽니다: {command.yaw_deg:+.3f} deg > ±{MAX_YAW_COMMAND_DEG:.1f} deg")
        return False

    return True


def within_tolerance(measurement: LeftMeasurement, error: PoseError) -> bool:
    """현재 측정이 실제 grasp 성공에 사용하던 위치/각도 허용 범위 안인지 판정한다."""
    position_ok = abs(error.x_m) <= POSITION_TOL_M and abs(error.y_m) <= POSITION_TOL_M
    angle_ok = abs(measurement.angle_deg - TARGET_ANGLE_DEG) <= IMAGE_ANGLE_TOL_DEG

    return position_ok and angle_ok


def trajectory_duration(command: RelativeCommand) -> float:
    """동시 x/y/yaw 이동에서 linear와 angular 속도 제한을 모두 만족하는 시간을 선택한다."""
    distance_m = math.hypot(command.x_m, command.y_m)
    yaw_rad = abs(math.radians(command.yaw_deg))

    linear_time = QUINTIC_PEAK * distance_m / ALIGN_LINEAR_SPEED if distance_m > 1e-8 else 0.0
    angular_time = QUINTIC_PEAK * yaw_rad / ALIGN_ANGULAR_SPEED if yaw_rad > 1e-8 else 0.0

    return max(linear_time, angular_time, MIN_LEG_TIME)


def move_one_shot(robot, monitor: OdometryMonitor, command: RelativeCommand) -> bool:
    """현재 자세 기준 x/y/yaw를 동시에 한 trajectory로 실행한다."""
    duration = trajectory_duration(command)

    print(f"trajectory duration: {duration:.2f} s")

    leg = build_leg(
        start=odom_pose(monitor.odom),
        target=(command.x_m, command.y_m, math.radians(command.yaw_deg)),
        absolute=False,
        duration=duration,
        turn_direction="shortest",
    )

    return move_leg(robot, monitor, leg, settle=SETTLE_S)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=None, help="사용할 D435 serial")
    parser.add_argument("--address", default=ADDRESS, help="RBY1 주소")
    parser.add_argument("--dry-run", action="store_true", help="계산/출력만 하고 실제 로봇은 움직이지 않음")
    args = parser.parse_args()

    pipeline = start_camera(args.serial)
    robot = None
    monitor = None

    try:
        if not args.dry_run:
            robot = initialize_mobile(address=args.address, model="m")
            monitor = OdometryMonitor()
            robot.start_state_update(monitor.on_state, rate=50)

            if not wait_for_odometry(monitor):
                print("Odometry를 받지 못했습니다.")
                return

        print()
        print("==============================================================")
        print("        TOTE ONE-SHOT ALIGN TEST V2")
        print("==============================================================")
        print(f"Target TL    : ({TARGET_TL_X_PX:.3f}, {TARGET_TL_Y_PX:.3f}) px")
        print(f"Target angle : {TARGET_ANGLE_DEG:+.3f} deg")
        print(f"Tolerance    : x/y ±{POSITION_TOL_M * 100:.1f} cm | image angle ±{IMAGE_ANGLE_TOL_DEG:.1f} deg")
        print("Feature      : TOP + LEFT corner only")
        print()

        flush_camera(pipeline)
        before = measure_left(pipeline, "BEFORE")

        if before is None:
            print("RESULT: FAIL - TOP + TL을 안정적으로 측정하지 못했습니다.")
            return

        before_error = estimate_pose_error(before)

        print()
        print_error("Estimated", before_error)

        if within_tolerance(before, before_error):
            print()
            print("이미 정답 범위 안입니다.")
            print("RESULT: SUCCESS")
            return

        command = pose_error_to_command(before_error)
        print_command(command)

        if not command_is_safe(command):
            print()
            print("RESULT: FAIL - local calibration 범위보다 command가 큽니다.")
            return

        if args.dry_run:
            print()
            print("DRY-RUN: 실제 이동은 수행하지 않았습니다.")
            return

        print()
        input("위 command로 x/y/yaw를 동시에 한 번 이동합니다. Enter > ")

        if not move_one_shot(robot, monitor, command):
            print("RESULT: FAIL - base 이동 실패")
            return

        time.sleep(SETTLE_S)
        flush_camera(pipeline)

        after = measure_left(pipeline, "AFTER")

        if after is None:
            print("RESULT: FAIL - 이동 후 TOP + TL 측정 실패")
            return

        after_error = estimate_pose_error(after)

        print()
        print_error("Final", after_error)
        print(f"Final image angle error: {abs(after.angle_deg - TARGET_ANGLE_DEG):.3f} deg")

        print()
        if within_tolerance(after, after_error):
            print("RESULT: SUCCESS")
        else:
            print("RESULT: FAIL")

    finally:
        if robot is not None:
            try:
                robot.stop_state_update()
            except Exception:
                pass

            robot.disconnect()

        pipeline.stop()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()