#!/usr/bin/env python3
"""LEFT/RIGHT corner를 독립 측정해 두 Yaw Jacobian을 자동 계산한다.

한쪽 corner가 화면에서 사라져도 보이는 쪽 측정값은 유지한다.
각 Yaw 자세의 측정 성공 여부와 관계없이 기준 heading으로 복귀한 뒤
반대 방향 측정을 계속한다.

출력
- TARGET_TL_X_PX / TARGET_TL_Y_PX / J_LEFT_YAW
- TARGET_TR_X_PX / TARGET_TR_Y_PX / J_RIGHT_YAW
- TARGET_ANGLE_DEG

실행:
python test/tote_yaw_auto_calibration_dual_v3.py --serial 250122079439
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
MIN_VALID_RATIO = 0.5
POSE_SETTLE_S = 1.5

SETTLE_S = 0.7
ALIGN_ANGULAR_SPEED = 0.5
QUINTIC_PEAK = 1.875
MIN_LEG_TIME = 1.5
MIN_RETURN_COMMAND_DEG = 0.05
MIN_JACOBIAN_SPAN_DEG = 0.2
COMMAND_STREAM_GAP_S = 1.5

WINDOW_NAME = "Tote DUAL Yaw Calibration v3"


@dataclass
class CornerMeasurement:
    """한 자세에서 한쪽 corner와 TOP angle의 통계."""

    x_px: float
    y_px: float
    angle_deg: float
    x_std_px: float
    y_std_px: float
    angle_std_deg: float
    valid_frames: int


@dataclass
class PoseMeasurement:
    """한 자세에서 독립적으로 측정한 TOP/LEFT/RIGHT 결과."""

    top_angle_deg: float | None
    top_angle_std_deg: float | None
    top_frames: int
    left: CornerMeasurement | None
    right: CornerMeasurement | None
    left_observed_frames: int
    right_observed_frames: int


@dataclass
class YawPoseSample:
    """실제 odom Yaw와 해당 자세 영상 측정값."""

    label: str
    yaw_deg: float
    measurement: PoseMeasurement


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

    if abs(yaw_error_deg) <= MIN_RETURN_COMMAND_DEG:
        print(f"기준 복귀 불필요: odom error={yaw_error_deg:+.4f} deg")
        return True

    print(f"기준 복귀 command: yaw={-yaw_error_deg:+.4f} deg")
    return move_relative_yaw(robot, monitor, -yaw_error_deg)


def prepare_measurement(pipeline: rs.pipeline) -> None:
    """모바일 정지 후 진동이 가라앉기를 기다리고 이동 중 frame을 버린다."""
    time.sleep(POSE_SETTLE_S)
    flush_camera(pipeline)


def build_corner_measurement(
    xs: list[float],
    ys: list[float],
    angles: list[float],
    min_valid_frames: int,
) -> CornerMeasurement | None:
    """충분한 frame이 있는 corner만 유효 측정값으로 만든다."""
    if len(xs) < min_valid_frames:
        return None

    return CornerMeasurement(
        x_px=float(np.median(xs)),
        y_px=float(np.median(ys)),
        angle_deg=float(np.median(angles)),
        x_std_px=float(np.std(xs)),
        y_std_px=float(np.std(ys)),
        angle_std_deg=float(np.std(angles)),
        valid_frames=len(xs),
    )


def measure_dual_feature(
    pipeline: rs.pipeline,
    label: str,
    frame_count: int,
) -> PoseMeasurement:
    """LEFT/RIGHT를 독립 수집한다. 한쪽 부재가 다른 쪽 측정을 실패시키지 않는다."""
    top_angles: list[float] = []
    left_xs: list[float] = []
    left_ys: list[float] = []
    left_angles: list[float] = []
    right_xs: list[float] = []
    right_ys: list[float] = []
    right_angles: list[float] = []
    start_time = time.monotonic()

    while time.monotonic() - start_time < MEASURE_TIMEOUT_S:
        if len(left_xs) >= frame_count and len(right_xs) >= frame_count:
            break

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

        angle_deg = float(feature.top_angle_deg)
        top_angles.append(angle_deg)

        if feature.left is not None and len(left_xs) < frame_count:
            left_xs.append(float(feature.left.point[0]))
            left_ys.append(float(feature.left.point[1]))
            left_angles.append(angle_deg)

        if feature.right is not None and len(right_xs) < frame_count:
            right_xs.append(float(feature.right.point[0]))
            right_ys.append(float(feature.right.point[1]))
            right_angles.append(angle_deg)

    min_valid_frames = max(1, math.ceil(frame_count * MIN_VALID_RATIO))
    left = build_corner_measurement(left_xs, left_ys, left_angles, min_valid_frames)
    right = build_corner_measurement(right_xs, right_ys, right_angles, min_valid_frames)

    measurement = PoseMeasurement(
        top_angle_deg=float(np.median(top_angles)) if top_angles else None,
        top_angle_std_deg=float(np.std(top_angles)) if top_angles else None,
        top_frames=len(top_angles),
        left=left,
        right=right,
        left_observed_frames=len(left_xs),
        right_observed_frames=len(right_xs),
    )
    print_pose_measurement(label, measurement, frame_count, min_valid_frames)
    return measurement


def corner_status(
    name: str,
    corner: CornerMeasurement | None,
    observed_frames: int,
    requested_frames: int,
    min_valid_frames: int,
) -> None:
    """한쪽 corner 통계 또는 부재 상태를 출력한다."""
    if corner is None:
        print(
            f"{name:<5}     : unavailable "
            f"({observed_frames}/{requested_frames}, minimum={min_valid_frames})"
        )
        return

    point_name = "TL" if name == "LEFT" else "TR"
    print(f"{point_name:<10}: ({corner.x_px:.3f}, {corner.y_px:.3f}) px")
    print(
        f"{name} std  : x={corner.x_std_px:.3f} px | y={corner.y_std_px:.3f} px | "
        f"angle={corner.angle_std_deg:.4f} deg | frames={corner.valid_frames}"
    )


def print_pose_measurement(
    label: str,
    measurement: PoseMeasurement,
    requested_frames: int,
    min_valid_frames: int,
) -> None:
    """한 자세의 TOP/LEFT/RIGHT 측정 결과를 출력한다."""
    print()
    print(f"[{label}]")

    if measurement.top_angle_deg is None:
        print("TOP angle : unavailable")
    else:
        print(
            f"TOP angle : {measurement.top_angle_deg:+.4f} deg | "
            f"std={measurement.top_angle_std_deg:.4f} deg | frames={measurement.top_frames}"
        )

    corner_status(
        "LEFT",
        measurement.left,
        measurement.left_observed_frames,
        requested_frames,
        min_valid_frames,
    )
    corner_status(
        "RIGHT",
        measurement.right,
        measurement.right_observed_frames,
        requested_frames,
        min_valid_frames,
    )

    if measurement.left is None and measurement.right is None:
        print("경고: 유효 corner가 없지만 기준 heading으로 복귀하고 다음 측정을 계속합니다.")


def measure_yaw_pose_and_return(
    robot,
    monitor: OdometryMonitor,
    pipeline: rs.pipeline,
    reference_yaw_rad: float,
    command_deg: float,
    label: str,
    frame_count: int,
) -> tuple[PoseMeasurement | None, float | None, bool]:
    """Yaw 이동/측정 후 측정 성공 여부와 무관하게 기준 heading으로 복귀한다."""
    print(f"\n{label}: command={command_deg:+.2f} deg")
    measurement = None
    actual_yaw_deg = None
    return_ok = False

    try:
        move_ok = move_relative_yaw(robot, monitor, command_deg)
        actual_yaw_deg = odom_yaw_delta_deg(monitor, reference_yaw_rad)

        if not move_ok:
            print(f"{label}: 이동 완료 판정 실패, actual odom={actual_yaw_deg:+.4f} deg")

        if abs(actual_yaw_deg) < MIN_JACOBIAN_SPAN_DEG:
            print(f"{label}: 실제 Yaw 변화가 너무 작아 영상 측정을 생략합니다.")
        else:
            prepare_measurement(pipeline)
            measurement = measure_dual_feature(pipeline, label, frame_count)
    finally:
        print(f"{label}: 기준 heading으로 복귀")
        return_ok = return_to_reference(robot, monitor, reference_yaw_rad)

        # move_leg()가 종료한 command stream이 서버에서 정리된 뒤
        # 다음 이동용 stream을 열도록 충분한 간격을 둔다.
        if return_ok:
            time.sleep(COMMAND_STREAM_GAP_S)

    final_error_deg = odom_yaw_delta_deg(monitor, reference_yaw_rad)
    print(f"{label}: 복귀 odom error={final_error_deg:+.4f} deg")
    return measurement, actual_yaw_deg, return_ok


def side_from_pose(measurement: PoseMeasurement, side: str) -> CornerMeasurement | None:
    """PoseMeasurement에서 지정 corner를 꺼낸다."""
    return measurement.left if side == "LEFT" else measurement.right


def regression_slope(yaws: np.ndarray, values: np.ndarray) -> float | None:
    """사용 가능한 두 개 이상의 자세로 value/yaw 최소제곱 기울기를 계산한다."""
    centered_yaws = yaws - float(np.mean(yaws))
    denominator = float(np.dot(centered_yaws, centered_yaws))

    if denominator < MIN_JACOBIAN_SPAN_DEG**2:
        return None

    centered_values = values - float(np.mean(values))
    return float(np.dot(centered_yaws, centered_values) / denominator)


def print_side_calibration(
    side: str,
    reference: PoseMeasurement,
    samples: list[YawPoseSample],
) -> None:
    """같은 corner가 보인 자세만 골라 해당 Yaw Jacobian을 계산한다."""
    usable: list[tuple[str, float, CornerMeasurement]] = []

    for sample in samples:
        corner = side_from_pose(sample.measurement, side)

        if corner is not None:
            usable.append((sample.label, sample.yaw_deg, corner))

    reference_corner = side_from_pose(reference, side)
    point_name = "TL" if side == "LEFT" else "TR"
    constant_prefix = "LEFT" if side == "LEFT" else "RIGHT"

    print()
    print(f"[{side} CALIBRATION]")
    print("usable poses: " + (", ".join(item[0] for item in usable) if usable else "none"))

    if reference_corner is None:
        print(f"REFERENCE에서 {point_name}이 없어 target을 만들 수 없습니다.")
    else:
        print(f"TARGET_{point_name}_X_PX = {reference_corner.x_px:.6f}")
        print(f"TARGET_{point_name}_Y_PX = {reference_corner.y_px:.6f}")
        print(f"TARGET_{constant_prefix}_ANGLE_DEG = {reference_corner.angle_deg:+.6f}")

    if len(usable) < 2:
        print(f"J_{constant_prefix}_YAW 계산 불가: 같은 corner가 보인 자세가 2개 미만입니다.")
        return

    yaws = np.asarray([item[1] for item in usable], dtype=np.float64)
    x_values = np.asarray([item[2].x_px for item in usable], dtype=np.float64)
    y_values = np.asarray([item[2].y_px for item in usable], dtype=np.float64)
    angle_values = np.asarray([item[2].angle_deg for item in usable], dtype=np.float64)

    d_x = regression_slope(yaws, x_values)
    d_y = regression_slope(yaws, y_values)
    d_angle = regression_slope(yaws, angle_values)

    if d_x is None or d_y is None or d_angle is None:
        print(f"J_{constant_prefix}_YAW 계산 불가: 실제 odom Yaw span이 너무 작습니다.")
        return

    print(f"J_{constant_prefix}_YAW = [{d_x:+.6f}, {d_y:+.6f}, {d_angle:+.6f}]")


def print_reference_drift(reference: PoseMeasurement, final_reference: PoseMeasurement) -> None:
    """최종 복귀 뒤 양쪽 corner와 TOP angle의 drift를 출력한다."""
    print()
    print("[FINAL REFERENCE DRIFT]")

    if reference.top_angle_deg is not None and final_reference.top_angle_deg is not None:
        print(f"TOP angle: {final_reference.top_angle_deg - reference.top_angle_deg:+.4f} deg")

    for side in ("LEFT", "RIGHT"):
        before = side_from_pose(reference, side)
        after = side_from_pose(final_reference, side)

        if before is None or after is None:
            print(f"{side}: drift 계산 불가")
            continue

        print(
            f"{side}: x={after.x_px - before.x_px:+.3f} px | "
            f"y={after.y_px - before.y_px:+.3f} px | "
            f"angle={after.angle_deg - before.angle_deg:+.4f} deg"
        )


def print_calibration_result(
    reference: PoseMeasurement,
    plus: PoseMeasurement | None,
    minus: PoseMeasurement | None,
    plus_yaw_deg: float | None,
    minus_yaw_deg: float | None,
    final_reference: PoseMeasurement | None,
    final_yaw_deg: float,
) -> None:
    """각 side에서 실제로 관측된 자세만 사용해 두 Yaw Jacobian을 출력한다."""
    samples = [YawPoseSample("REFERENCE", 0.0, reference)]

    if plus is not None and plus_yaw_deg is not None:
        samples.append(YawPoseSample("+YAW", plus_yaw_deg, plus))

    if minus is not None and minus_yaw_deg is not None:
        samples.append(YawPoseSample("-YAW", minus_yaw_deg, minus))

    print()
    print("=" * 96)
    print("DUAL YAW CALIBRATION RESULT")
    print("=" * 96)

    if plus_yaw_deg is not None:
        print(f"+Yaw actual odom : {plus_yaw_deg:+.4f} deg")

    if minus_yaw_deg is not None:
        print(f"-Yaw actual odom : {minus_yaw_deg:+.4f} deg")

    print(f"Final odom error : {final_yaw_deg:+.4f} deg")

    if reference.top_angle_deg is not None:
        print(f"TARGET_ANGLE_DEG = {reference.top_angle_deg:+.6f}")

    print_side_calibration("LEFT", reference, samples)
    print_side_calibration("RIGHT", reference, samples)

    if final_reference is not None:
        print_reference_drift(reference, final_reference)

    print()
    print("Runtime 선택 규칙: RIGHT가 보이면 RIGHT, 아니면 LEFT가 보이면 LEFT, 둘 다 없으면 recovery")
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tote LEFT/RIGHT Yaw 자동 calibration v3")
    parser.add_argument("--serial", default=None, help="사용할 D435 serial")
    parser.add_argument("--yaw-deg", type=float, default=YAW_TEST_DEG, help="자동 회전 크기")
    parser.add_argument("--frames", type=int, default=MEASURE_FRAMES, help="각 자세의 corner별 목표 frame 수")
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
        print("       TOTE DUAL AUTO YAW CALIBRATION v3")
        print("==============================================================")
        print(f"Yaw command  : ±{args.yaw_deg:.2f} deg")
        print(f"Corner frames: {args.frames} each")
        print(f"Valid minimum: {math.ceil(args.frames * MIN_VALID_RATIO)} each")
        print()
        print("로봇을 실제 grasp 성공 기준 자세에 놓아주세요.")
        print("한쪽 corner가 사라져도 보이는 쪽을 저장하고 기준 복귀 후 계속 진행합니다.")
        input("준비되면 Enter > ")

        reference_yaw_rad = odom_pose(monitor.odom)[2]
        prepare_measurement(pipeline)
        reference = measure_dual_feature(pipeline, "REFERENCE", args.frames)

        if reference.left is None and reference.right is None:
            print("REFERENCE에서 유효한 LEFT/RIGHT corner가 모두 없어 calibration을 시작할 수 없습니다.")
            return

        plus, plus_yaw_deg, plus_return_ok = measure_yaw_pose_and_return(
            robot,
            monitor,
            pipeline,
            reference_yaw_rad,
            +args.yaw_deg,
            "+YAW",
            args.frames,
        )

        if not plus_return_ok:
            print("+YAW 후 기준 복귀 실패: 안전을 위해 반대 방향 측정을 중단합니다.")
            return

        minus, minus_yaw_deg, minus_return_ok = measure_yaw_pose_and_return(
            robot,
            monitor,
            pipeline,
            reference_yaw_rad,
            -args.yaw_deg,
            "-YAW",
            args.frames,
        )

        if not minus_return_ok:
            print("-YAW 후 기준 복귀 실패: 결과를 출력한 뒤 종료합니다.")
            final_yaw_deg = odom_yaw_delta_deg(monitor, reference_yaw_rad)
            final_reference = None
        else:
            prepare_measurement(pipeline)
            final_yaw_deg = odom_yaw_delta_deg(monitor, reference_yaw_rad)
            final_reference = measure_dual_feature(pipeline, "FINAL REFERENCE", args.frames)

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
