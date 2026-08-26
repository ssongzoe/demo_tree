#!/usr/bin/env python3
"""TOP + TL/TR의 X/Y translation Jacobian을 자동으로 측정하는 테스트.

동작 순서
- REFERENCE 측정
- +X 2 cm -> 측정 -> 기준 복귀
- -X 2 cm -> 측정 -> 기준 복귀
- +Y 2 cm -> 측정 -> 기준 복귀
- -Y 2 cm -> 측정 -> 기준 복귀
- 실제 odom displacement를 사용해 LEFT / RIGHT translation Jacobian을 least-squares로 계산

중요
- 사람이 +X/-X/+Y/-Y 방향을 정하지 않는다.
- mobile_controller의 실제 로봇 좌표계를 그대로 사용한다.
- 각 샘플의 실제 odom dx/dy/yaw를 사용한다.
- 작은 yaw drift는 이미 측정한 yaw Jacobian column으로 보정한 뒤 X/Y column을 계산한다.

실행:
python test/tote_xy_auto_calibration_test.py --serial 250122079439
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


ADDRESS = "192.168.30.1:50051"

XY_TEST_M = 0.02

SETTLE_S = 0.7
ALIGN_LINEAR_SPEED = 0.08
ALIGN_ANGULAR_SPEED = 0.5
QUINTIC_PEAK = 1.875
MIN_LEG_TIME = 1.5

MEASURE_FRAMES = 40
MEASURE_TIMEOUT_S = 8.0
CAMERA_FLUSH_FRAMES = 10

WINDOW_NAME = "Tote XY Auto Calibration"

# 이전 자동 yaw calibration 3회 평균값
# 단위: [px/deg, px/deg, deg/deg]
J_LEFT_YAW = np.asarray([+14.718822, -3.116924, +0.712066], dtype=np.float64)

# RIGHT는 yaw에서 LEFT보다 흔들림이 컸으므로 참고용
J_RIGHT_YAW = np.asarray([+15.296616, +2.911850, +0.712066], dtype=np.float64)


@dataclass
class CalibrationMeasurement:
    angle_deg: float
    tl_x: float | None
    tl_y: float | None
    tr_x: float | None
    tr_y: float | None
    top_frames: int
    left_frames: int
    right_frames: int


@dataclass
class PoseOffset:
    x_m: float
    y_m: float
    yaw_deg: float


@dataclass
class CalibrationSample:
    label: str
    pose: PoseOffset
    measurement: CalibrationMeasurement


def wrap_angle_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def relative_pose(reference_pose: tuple[float, float, float], current_pose: tuple[float, float, float]) -> PoseOffset:
    ref_x, ref_y, ref_yaw = reference_pose
    cur_x, cur_y, cur_yaw = current_pose

    dx_world = cur_x - ref_x
    dy_world = cur_y - ref_y

    cos_ref = math.cos(ref_yaw)
    sin_ref = math.sin(ref_yaw)

    dx_local = cos_ref * dx_world + sin_ref * dy_world
    dy_local = -sin_ref * dx_world + cos_ref * dy_world
    dyaw_deg = math.degrees(wrap_angle_rad(cur_yaw - ref_yaw))

    return PoseOffset(x_m=dx_local, y_m=dy_local, yaw_deg=dyaw_deg)


def trajectory_duration(x_m: float, y_m: float, yaw_deg: float = 0.0) -> float:
    distance_m = math.hypot(x_m, y_m)
    yaw_rad = abs(math.radians(yaw_deg))

    linear_time = QUINTIC_PEAK * distance_m / ALIGN_LINEAR_SPEED if distance_m > 1e-8 else 0.0
    angular_time = QUINTIC_PEAK * yaw_rad / ALIGN_ANGULAR_SPEED if yaw_rad > 1e-8 else 0.0

    return max(linear_time, angular_time, MIN_LEG_TIME)


def move_relative(robot, monitor: OdometryMonitor, x_m: float = 0.0, y_m: float = 0.0, yaw_deg: float = 0.0) -> bool:
    duration = trajectory_duration(x_m, y_m, yaw_deg)

    leg = build_leg(
        start=odom_pose(monitor.odom),
        target=(x_m, y_m, math.radians(yaw_deg)),
        absolute=False,
        duration=duration,
        turn_direction="shortest",
    )

    return move_leg(robot, monitor, leg, settle=SETTLE_S)


def return_to_reference(robot, monitor: OdometryMonitor, reference_pose: tuple[float, float, float]) -> bool:
    error = relative_pose(reference_pose, odom_pose(monitor.odom))

    yaw_rad = math.radians(error.yaw_deg)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    command_x = -(cos_yaw * error.x_m + sin_yaw * error.y_m)
    command_y = +(sin_yaw * error.x_m - cos_yaw * error.y_m)
    command_yaw = -error.yaw_deg

    print(f"기준 복귀 command: x={command_x:+.4f} m | y={command_y:+.4f} m | yaw={command_yaw:+.3f} deg")

    return move_relative(robot, monitor, x_m=command_x, y_m=command_y, yaw_deg=command_yaw)


def flush_camera(pipeline: rs.pipeline) -> None:
    for _ in range(CAMERA_FLUSH_FRAMES):
        pipeline.wait_for_frames()


def measure_pose(pipeline: rs.pipeline, label: str, frame_count: int) -> CalibrationMeasurement | None:
    angles: list[float] = []
    tl_xs: list[float] = []
    tl_ys: list[float] = []
    tr_xs: list[float] = []
    tr_ys: list[float] = []

    start_time = time.monotonic()

    while len(angles) < frame_count and time.monotonic() - start_time < MEASURE_TIMEOUT_S:
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

        if feature is None:
            continue

        angles.append(float(feature.top_angle_deg))

        if feature.left is not None:
            tl_xs.append(float(feature.left.point[0]))
            tl_ys.append(float(feature.left.point[1]))

        if feature.right is not None:
            tr_xs.append(float(feature.right.point[0]))
            tr_ys.append(float(feature.right.point[1]))

    if len(angles) < frame_count // 2:
        print(f"{label}: TOP 측정 실패 ({len(angles)}/{frame_count})")
        return None

    result = CalibrationMeasurement(
        angle_deg=float(np.median(angles)),
        tl_x=float(np.median(tl_xs)) if tl_xs else None,
        tl_y=float(np.median(tl_ys)) if tl_ys else None,
        tr_x=float(np.median(tr_xs)) if tr_xs else None,
        tr_y=float(np.median(tr_ys)) if tr_ys else None,
        top_frames=len(angles),
        left_frames=len(tl_xs),
        right_frames=len(tr_xs),
    )

    print_measurement(label, result)
    return result


def print_measurement(label: str, measurement: CalibrationMeasurement) -> None:
    tl_text = "N/A" if measurement.tl_x is None else f"({measurement.tl_x:.3f}, {measurement.tl_y:.3f})"
    tr_text = "N/A" if measurement.tr_x is None else f"({measurement.tr_x:.3f}, {measurement.tr_y:.3f})"

    print()
    print(f"[{label}]")
    print(f"TOP angle : {measurement.angle_deg:+.4f} deg")
    print(f"TL        : {tl_text}   frames={measurement.left_frames}")
    print(f"TR        : {tr_text}   frames={measurement.right_frames}")


def print_pose(label: str, pose: PoseOffset) -> None:
    print(f"{label} odom offset: x={pose.x_m:+.4f} m | y={pose.y_m:+.4f} m | yaw={pose.yaw_deg:+.4f} deg")


def collect_sample(
    robot,
    monitor: OdometryMonitor,
    pipeline: rs.pipeline,
    reference_pose: tuple[float, float, float],
    label: str,
    x_m: float,
    y_m: float,
    frame_count: int,
) -> CalibrationSample | None:
    print()
    print("=" * 90)
    print(f"{label}: command x={x_m:+.4f} m | y={y_m:+.4f} m")
    print("=" * 90)

    if not move_relative(robot, monitor, x_m=x_m, y_m=y_m):
        print(f"{label}: 이동 실패")
        return None

    flush_camera(pipeline)

    pose = relative_pose(reference_pose, odom_pose(monitor.odom))
    print_pose(label, pose)

    measurement = measure_pose(pipeline, label, frame_count)

    if measurement is None:
        return None

    return CalibrationSample(label=label, pose=pose, measurement=measurement)


def fit_translation_jacobian(
    reference: CalibrationMeasurement,
    samples: list[CalibrationSample],
    side: str,
    yaw_column: np.ndarray,
) -> np.ndarray | None:
    rows = []
    targets = []

    if side == "LEFT":
        if reference.tl_x is None:
            return None

        reference_feature = np.asarray([reference.tl_x, reference.tl_y, reference.angle_deg], dtype=np.float64)

        for sample in samples:
            m = sample.measurement

            if m.tl_x is None:
                continue

            feature = np.asarray([m.tl_x, m.tl_y, m.angle_deg], dtype=np.float64)
            delta = feature - reference_feature
            delta -= yaw_column * sample.pose.yaw_deg

            rows.append([sample.pose.x_m, sample.pose.y_m])
            targets.append(delta)

    elif side == "RIGHT":
        if reference.tr_x is None:
            return None

        reference_feature = np.asarray([reference.tr_x, reference.tr_y, reference.angle_deg], dtype=np.float64)

        for sample in samples:
            m = sample.measurement

            if m.tr_x is None:
                continue

            feature = np.asarray([m.tr_x, m.tr_y, m.angle_deg], dtype=np.float64)
            delta = feature - reference_feature
            delta -= yaw_column * sample.pose.yaw_deg

            rows.append([sample.pose.x_m, sample.pose.y_m])
            targets.append(delta)

    else:
        raise ValueError(f"지원하지 않는 side: {side}")

    if len(rows) < 2:
        return None

    A = np.asarray(rows, dtype=np.float64)
    B = np.asarray(targets, dtype=np.float64)

    if np.linalg.matrix_rank(A) < 2:
        print(f"{side}: calibration odom sample rank가 부족합니다.")
        return None

    coefficients, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    return coefficients.T


def print_jacobian(side: str, translation: np.ndarray | None, yaw_column: np.ndarray) -> None:
    if translation is None:
        print(f"{side}: Jacobian 계산 불가")
        return

    full = np.column_stack([translation, yaw_column])

    print()
    print("=" * 90)
    print(f"{side} CALIBRATION RESULT")
    print("=" * 90)
    print("row    = [corner_x(px), corner_y(px), top_angle(deg)]")
    print("column = [robot_x(m), robot_y(m), robot_yaw(deg)]")
    print()

    print(f"J_{side}_X = [{translation[0, 0]:+.6f}, {translation[1, 0]:+.6f}, {translation[2, 0]:+.6f}]")
    print(f"J_{side}_Y = [{translation[0, 1]:+.6f}, {translation[1, 1]:+.6f}, {translation[2, 1]:+.6f}]")
    print(f"J_{side}_YAW = [{yaw_column[0]:+.6f}, {yaw_column[1]:+.6f}, {yaw_column[2]:+.6f}]")
    print()

    print(f"J_{side} = np.asarray(")
    print("    [")
    print(f"        [{full[0, 0]:+10.4f}, {full[0, 1]:+10.4f}, {full[0, 2]:+10.4f}],")
    print(f"        [{full[1, 0]:+10.4f}, {full[1, 1]:+10.4f}, {full[1, 2]:+10.4f}],")
    print(f"        [{full[2, 0]:+10.4f}, {full[2, 1]:+10.4f}, {full[2, 2]:+10.4f}],")
    print("    ],")
    print("    dtype=np.float64,")
    print(")")

    if side == "LEFT":
        inverse = np.linalg.inv(full)

        print()
        print("J_LEFT_INV = np.asarray(")
        print("    [")
        print(f"        [{inverse[0, 0]:+12.8f}, {inverse[0, 1]:+12.8f}, {inverse[0, 2]:+12.8f}],")
        print(f"        [{inverse[1, 0]:+12.8f}, {inverse[1, 1]:+12.8f}, {inverse[1, 2]:+12.8f}],")
        print(f"        [{inverse[2, 0]:+12.8f}, {inverse[2, 1]:+12.8f}, {inverse[2, 2]:+12.8f}],")
        print("    ],")
        print("    dtype=np.float64,")
        print(")")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=None, help="사용할 D435 serial")
    parser.add_argument("--distance", type=float, default=XY_TEST_M, help="자동 X/Y calibration 이동 거리(m)")
    parser.add_argument("--frames", type=int, default=MEASURE_FRAMES, help="각 자세에서 사용할 TOP frame 수")
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
        print("        TOTE AUTO X/Y CALIBRATION")
        print("==============================================================")
        print(f"Translation command : ±{args.distance * 100:.1f} cm")
        print(f"Frames              : {args.frames}")
        print()
        print("로봇을 grasp 성공 기준 자세에 놓아주세요.")
        print("Enter 후 +X / -X / +Y / -Y를 자동으로 움직이고 매번 기준 자세로 복귀합니다.")
        input("준비되면 Enter > ")

        flush_camera(pipeline)

        reference = measure_pose(pipeline, "REFERENCE", args.frames)

        if reference is None:
            return

        reference_pose = odom_pose(monitor.odom)
        samples: list[CalibrationSample] = []

        tests = [
            ("+X", +args.distance, 0.0),
            ("-X", -args.distance, 0.0),
            ("+Y", 0.0, +args.distance),
            ("-Y", 0.0, -args.distance),
        ]

        for index, (label, x_m, y_m) in enumerate(tests, start=1):
            sample = collect_sample(robot, monitor, pipeline, reference_pose, label, x_m, y_m, args.frames)

            if sample is None:
                print(f"{label} calibration 실패")
                return

            samples.append(sample)

            print(f"\n[{index}/4] {label} 측정 완료 -> 기준 자세 복귀")

            if not return_to_reference(robot, monitor, reference_pose):
                print("기준 자세 복귀 실패")
                return

            flush_camera(pipeline)
            return_error = relative_pose(reference_pose, odom_pose(monitor.odom))
            print_pose("RETURN", return_error)

        print()
        print("최종 reference 영상 재측정")
        final_reference = measure_pose(pipeline, "FINAL REFERENCE", args.frames)

        if final_reference is not None:
            if reference.tl_x is not None and final_reference.tl_x is not None:
                print(
                    f"LEFT reference drift: dx={final_reference.tl_x - reference.tl_x:+.3f} px | "
                    f"dy={final_reference.tl_y - reference.tl_y:+.3f} px | "
                    f"angle={final_reference.angle_deg - reference.angle_deg:+.4f} deg"
                )

            if reference.tr_x is not None and final_reference.tr_x is not None:
                print(
                    f"RIGHT reference drift: dx={final_reference.tr_x - reference.tr_x:+.3f} px | "
                    f"dy={final_reference.tr_y - reference.tr_y:+.3f} px"
                )

        left_translation = fit_translation_jacobian(reference, samples, "LEFT", J_LEFT_YAW)
        right_translation = fit_translation_jacobian(reference, samples, "RIGHT", J_RIGHT_YAW)

        print_jacobian("LEFT", left_translation, J_LEFT_YAW)
        print_jacobian("RIGHT", right_translation, J_RIGHT_YAW)

    finally:
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
