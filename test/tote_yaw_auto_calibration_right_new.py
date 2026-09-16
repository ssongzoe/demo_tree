#!/usr/bin/env python3
"""운영용 Robust detector로 RIGHT feature의 Yaw Jacobian을 자동 측정한다.

동작 순서
- REFERENCE에서 [TR.x, TR.y, TOP angle] 측정
- +Yaw 회전 후 측정
- 기준 heading 복귀
- -Yaw 회전 후 측정
- 기준 heading 복귀 및 최종 drift 측정
- 실제 odom 회전량으로 J_RIGHT_YAW 계산

실행:
python test/tote_yaw_auto_calibration_right_new.py --serial 250122079439
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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.mobile_controller import (
    OdometryMonitor,
    build_leg,
    initialize_mobile,
    move_leg,
    odom_pose,
    wait_for_odometry,
)
from utils.tote_vision import detect_frame_feature, draw_feature, flush_camera, start_camera


ADDRESS = "192.168.30.1:50051"

YAW_TEST_DEG = 3.0
MEASURE_FRAMES = 80
MEASURE_TIMEOUT_S = 8.0
POSE_SETTLE_S = 1.5

SETTLE_S = 0.7
ALIGN_ANGULAR_SPEED = 0.5
QUINTIC_PEAK = 1.875
MIN_LEG_TIME = 1.5

WINDOW_NAME = "Tote RIGHT Yaw Calibration"


@dataclass
class RightMeasurement:
    """한 자세에서 여러 frame으로 얻은 RIGHT feature median."""

    tr_x_px: float
    tr_y_px: float
    angle_deg: float
    tr_x_std_px: float
    tr_y_std_px: float
    angle_std_deg: float
    top_frames: int
    right_frames: int


def wrap_angle_rad(angle: float) -> float:
    """각도를 -pi~pi 범위로 정규화한다."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def turn_duration(angle_deg: float) -> float:
    """Yaw 크기에 맞는 trajectory 시간을 계산한다."""
    angle_rad = abs(math.radians(angle_deg))
    return max(QUINTIC_PEAK * angle_rad / ALIGN_ANGULAR_SPEED, MIN_LEG_TIME)


def move_relative_yaw(robot, monitor: OdometryMonitor, angle_deg: float) -> bool:
    """현재 자세 기준으로 Yaw만 상대 이동한다."""
    leg = build_leg(
        start=odom_pose(monitor.odom),
        target=(0.0, 0.0, math.radians(angle_deg)),
        absolute=False,
        duration=turn_duration(angle_deg),
        turn_direction="shortest",
    )

    return move_leg(robot, monitor, leg, settle=SETTLE_S)


def odom_yaw_delta_deg(monitor: OdometryMonitor, reference_yaw_rad: float) -> float:
    """기준 heading으로부터 현재 odom Yaw 차이를 반환한다."""
    current_yaw_rad = odom_pose(monitor.odom)[2]
    return math.degrees(wrap_angle_rad(current_yaw_rad - reference_yaw_rad))


def return_to_reference(robot, monitor: OdometryMonitor, reference_yaw_rad: float) -> bool:
    """현재 odom Yaw 오차만큼 반대로 회전해 기준 heading으로 복귀한다."""
    yaw_error_deg = odom_yaw_delta_deg(monitor, reference_yaw_rad)
    print(f"기준 복귀 command: yaw={-yaw_error_deg:+.4f} deg")
    return move_relative_yaw(robot, monitor, -yaw_error_deg)


def prepare_measurement(pipeline: rs.pipeline) -> None:
    """모바일 정지 후 진동이 가라앉기를 기다리고 이동 중 frame을 버린다."""
    time.sleep(POSE_SETTLE_S)
    flush_camera(pipeline)


def measure_right_feature(
    pipeline: rs.pipeline,
    label: str,
    frame_count: int,
) -> RightMeasurement | None:
    """TR이 검출된 frame을 모아 RIGHT feature median과 std를 반환한다."""
    tr_xs: list[float] = []
    tr_ys: list[float] = []
    angles: list[float] = []
    top_frames = 0
    start_time = time.monotonic()

    while len(tr_xs) < frame_count and time.monotonic() - start_time < MEASURE_TIMEOUT_S:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asarray(color_frame.get_data())
        feature = detect_frame_feature(image)
        cv2.imshow(WINDOW_NAME, draw_feature(image, feature, label=label))
        cv2.waitKey(1)

        if feature is None:
            continue

        top_frames += 1

        if feature.right is None:
            continue

        tr_xs.append(float(feature.right.point[0]))
        tr_ys.append(float(feature.right.point[1]))
        angles.append(float(feature.top_angle_deg))

    if len(tr_xs) < frame_count // 2:
        print(f"{label}: RIGHT 측정 실패 (TOP={top_frames}, RIGHT={len(tr_xs)}/{frame_count})")
        return None

    measurement = RightMeasurement(
        tr_x_px=float(np.median(tr_xs)),
        tr_y_px=float(np.median(tr_ys)),
        angle_deg=float(np.median(angles)),
        tr_x_std_px=float(np.std(tr_xs)),
        tr_y_std_px=float(np.std(tr_ys)),
        angle_std_deg=float(np.std(angles)),
        top_frames=top_frames,
        right_frames=len(tr_xs),
    )
    print_measurement(label, measurement)
    return measurement


def print_measurement(label: str, measurement: RightMeasurement) -> None:
    """현재 자세의 RIGHT feature 측정값을 출력한다."""
    print()
    print(f"[{label}]")
    print(f"TR        : ({measurement.tr_x_px:.3f}, {measurement.tr_y_px:.3f}) px")
    print(f"TOP angle : {measurement.angle_deg:+.4f} deg")
    print(
        f"std       : TR.x={measurement.tr_x_std_px:.3f} px | "
        f"TR.y={measurement.tr_y_std_px:.3f} px | angle={measurement.angle_std_deg:.4f} deg"
    )
    print(f"frames    : TOP={measurement.top_frames} | RIGHT={measurement.right_frames}")


def print_calibration_result(
    reference: RightMeasurement,
    plus: RightMeasurement,
    minus: RightMeasurement,
    plus_yaw_deg: float,
    minus_yaw_deg: float,
    final_reference: RightMeasurement | None,
    final_yaw_deg: float,
) -> None:
    """+/- Yaw 측정값으로 RIGHT feature의 Yaw column을 계산한다."""
    robot_span_deg = plus_yaw_deg - minus_yaw_deg

    if abs(robot_span_deg) < 1e-6:
        print("실제 odom Yaw span이 너무 작아서 결과를 계산할 수 없습니다.")
        return

    d_tr_x = (plus.tr_x_px - minus.tr_x_px) / robot_span_deg
    d_tr_y = (plus.tr_y_px - minus.tr_y_px) / robot_span_deg
    d_angle = (plus.angle_deg - minus.angle_deg) / robot_span_deg

    print()
    print("=" * 94)
    print("RIGHT YAW CALIBRATION RESULT")
    print("=" * 94)
    print(f"+Yaw actual odom : {plus_yaw_deg:+.4f} deg")
    print(f"-Yaw actual odom : {minus_yaw_deg:+.4f} deg")
    print(f"Robot yaw span   : {robot_span_deg:+.4f} deg")
    print(f"Final odom error : {final_yaw_deg:+.4f} deg")
    print()
    print(f"TARGET_TR_X_PX   = {reference.tr_x_px:.6f}")
    print(f"TARGET_TR_Y_PX   = {reference.tr_y_px:.6f}")
    print(f"TARGET_ANGLE_DEG = {reference.angle_deg:+.6f}")
    print(f"J_RIGHT_YAW = [{d_tr_x:+.6f}, {d_tr_y:+.6f}, {d_angle:+.6f}]")

    if final_reference is not None:
        print()
        print(
            f"Reference drift: TR.x={final_reference.tr_x_px - reference.tr_x_px:+.3f} px | "
            f"TR.y={final_reference.tr_y_px - reference.tr_y_px:+.3f} px | "
            f"angle={final_reference.angle_deg - reference.angle_deg:+.4f} deg"
        )

    print("=" * 94)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tote RIGHT Yaw 자동 calibration")
    parser.add_argument("--serial", default=None, help="사용할 D435 serial")
    parser.add_argument("--yaw-deg", type=float, default=YAW_TEST_DEG, help="자동 회전 크기")
    parser.add_argument("--frames", type=int, default=MEASURE_FRAMES, help="각 자세의 유효 RIGHT frame 수")
    parser.add_argument("--address", default=ADDRESS, help="RBY1 주소")
    args = parser.parse_args()

    pipeline = start_camera(args.serial)
    robot = initialize_mobile(address=args.address, model="m")
    monitor = OdometryMonitor()

    try:
        robot.start_state_update(monitor.on_state, rate=50)

        if not wait_for_odometry(monitor):
            print("Odometry를 받지 못했습니다.")
            return

        print()
        print("==============================================================")
        print("       TOTE RIGHT AUTO YAW CALIBRATION")
        print("==============================================================")
        print(f"Yaw command : ±{args.yaw_deg:.2f} deg")
        print(f"RIGHT frames: {args.frames}")
        print()
        print("로봇을 실제 grasp 성공 기준 자세에 놓아주세요.")
        print("Enter 후 REFERENCE → +Yaw → 복귀 → -Yaw → 복귀 순서로 자동 진행합니다.")
        input("준비되면 Enter > ")

        prepare_measurement(pipeline)
        reference = measure_right_feature(pipeline, "REFERENCE", args.frames)

        if reference is None:
            return

        reference_yaw_rad = odom_pose(monitor.odom)[2]

        print(f"\n[1/4] +{args.yaw_deg:.2f} deg 회전")
        if not move_relative_yaw(robot, monitor, +args.yaw_deg):
            print("+Yaw 이동 실패")
            return

        prepare_measurement(pipeline)
        plus_yaw_deg = odom_yaw_delta_deg(monitor, reference_yaw_rad)
        plus = measure_right_feature(pipeline, "+YAW", args.frames)

        if plus is None:
            return

        print("\n[2/4] 기준 heading으로 복귀")
        if not return_to_reference(robot, monitor, reference_yaw_rad):
            print("기준 복귀 실패")
            return

        prepare_measurement(pipeline)
        print(f"기준 복귀 odom error: {odom_yaw_delta_deg(monitor, reference_yaw_rad):+.4f} deg")

        print(f"\n[3/4] -{args.yaw_deg:.2f} deg 회전")
        if not move_relative_yaw(robot, monitor, -args.yaw_deg):
            print("-Yaw 이동 실패")
            return

        prepare_measurement(pipeline)
        minus_yaw_deg = odom_yaw_delta_deg(monitor, reference_yaw_rad)
        minus = measure_right_feature(pipeline, "-YAW", args.frames)

        if minus is None:
            return

        print("\n[4/4] 기준 heading으로 복귀")
        if not return_to_reference(robot, monitor, reference_yaw_rad):
            print("최종 복귀 실패")
            return

        prepare_measurement(pipeline)
        final_yaw_deg = odom_yaw_delta_deg(monitor, reference_yaw_rad)
        final_reference = measure_right_feature(pipeline, "FINAL REFERENCE", args.frames)

        print_calibration_result(
            reference,
            plus,
            minus,
            plus_yaw_deg,
            minus_yaw_deg,
            final_reference,
            final_yaw_deg,
        )

    finally:
        try:
            robot.stop_state_update()
        except Exception:
            pass

        robot.disconnect()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
