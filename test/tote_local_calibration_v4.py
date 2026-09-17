#!/usr/bin/env python3
"""RB-Y1 Tote local 3-DoF calibration.

Run this script with the robot placed at the exact desired grasp pose.
The script measures the reference, visits +/-X, +/-Y and +/-Yaw poses,
returns to the reference after every pose, and fits the complete 3x3 image
Jacobian from actual odometry.

Feature vectors:
    LEFT   = [TL.x(px), TL.y(px), top_angle(deg)]
    RIGHT  = [TR.x(px), TR.y(px), top_angle(deg)]
    CENTER = [rim_center_x(px), rim_center_y(px), top_angle(deg)]

The saved JSON and printed constants are inputs for the next coupled aligner.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np


# This also lets the file run from demo_tree/test/ without PYTHONPATH setup.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "control").is_dir() else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.mobile_controller import (  # noqa: E402
    OdometryMonitor,
    build_leg,
    initialize_mobile,
    move_leg,
    odom_pose,
    wait_for_odometry,
)
from control.robot_controller import move_both_arms, move_torso_and_head  # noqa: E402
from utils.ar_marker import RealSenseCamera  # noqa: E402
from utils.tote_vision import detect_frame_feature, draw_feature, flush_camera  # noqa: E402


WINDOW_NAME = "Tote Local Calibration v4"


@dataclass
class FrameSample:
    angle_deg: float
    left_x: float | None
    left_y: float | None
    right_x: float | None
    right_y: float | None


@dataclass
class WindowMeasurement:
    label: str
    total_frames: int
    cluster_frames: int
    raw_angle_std_deg: float
    angle_mad_deg: float
    left_frames: int
    right_frames: int
    center_frames: int
    left: np.ndarray | None
    right: np.ndarray | None
    center: np.ndarray | None


def wrap_angle_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def pose_delta(reference, current) -> np.ndarray:
    """Current pose minus reference, expressed in the reference body frame."""
    ref_x, ref_y, ref_yaw = reference
    cur_x, cur_y, cur_yaw = current
    dx_world = cur_x - ref_x
    dy_world = cur_y - ref_y
    cos_yaw = math.cos(ref_yaw)
    sin_yaw = math.sin(ref_yaw)
    return np.asarray(
        [
            cos_yaw * dx_world + sin_yaw * dy_world,
            -sin_yaw * dx_world + cos_yaw * dy_world,
            math.degrees(wrap_angle_rad(cur_yaw - ref_yaw)),
        ],
        dtype=np.float64,
    )


def offset_world_pose(reference, offset) -> tuple[float, float, float]:
    """Apply [body_x(m), body_y(m), yaw(deg)] to a world-frame pose."""
    ref_x, ref_y, ref_yaw = reference
    body_x, body_y, yaw_deg = offset
    cos_yaw = math.cos(ref_yaw)
    sin_yaw = math.sin(ref_yaw)
    return (
        ref_x + cos_yaw * body_x - sin_yaw * body_y,
        ref_y + sin_yaw * body_x + cos_yaw * body_y,
        ref_yaw + math.radians(yaw_deg),
    )


def relative_command(current, target) -> tuple[float, float, float]:
    """World target converted to current body-frame [x, y, yaw(rad)]."""
    cur_x, cur_y, cur_yaw = current
    target_x, target_y, target_yaw = target
    dx_world = target_x - cur_x
    dy_world = target_y - cur_y
    cos_yaw = math.cos(cur_yaw)
    sin_yaw = math.sin(cur_yaw)
    return (
        cos_yaw * dx_world + sin_yaw * dy_world,
        -sin_yaw * dx_world + cos_yaw * dy_world,
        wrap_angle_rad(target_yaw - cur_yaw),
    )


def trajectory_duration(command) -> float:
    x_m, y_m, yaw_rad = command
    translation_m = math.hypot(x_m, y_m)
    linear_time = 1.875 * translation_m / 0.06
    angular_time = 1.875 * abs(yaw_rad) / 0.35
    return max(1.5, linear_time, angular_time)


def move_to_pose(robot, monitor: OdometryMonitor, target, label: str) -> bool:
    """Move to a world pose and make at most one small residual correction."""
    for attempt in range(2):
        current = odom_pose(monitor.odom)
        command = relative_command(current, target)
        translation_m = math.hypot(command[0], command[1])
        yaw_deg = math.degrees(command[2])

        if translation_m <= 0.003 and abs(yaw_deg) <= 0.08:
            return True

        print(
            f"{label} ({attempt + 1}/2): "
            f"x={command[0]:+.4f} m, y={command[1]:+.4f} m, "
            f"yaw={yaw_deg:+.3f} deg"
        )
        leg = build_leg(
            start=current,
            target=command,
            absolute=False,
            duration=trajectory_duration(command),
            turn_direction="shortest",
        )
        if not move_leg(robot, monitor, leg, settle=0.8):
            return False

    final_command = relative_command(odom_pose(monitor.odom), target)
    final_translation = math.hypot(final_command[0], final_command[1])
    final_yaw_deg = math.degrees(final_command[2])
    print(
        f"{label} residual: translation={final_translation * 100:.2f} cm, "
        f"yaw={final_yaw_deg:+.3f} deg"
    )
    return final_translation <= 0.008 and abs(final_yaw_deg) <= 0.20


def median_mad(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return median, mad


def dominant_angle_mask(
    samples: list[FrameSample],
    radius_deg: float,
) -> np.ndarray:
    """Select the densest 1-D angle mode instead of mixing two line hypotheses."""
    angles = np.asarray([sample.angle_deg for sample in samples], dtype=np.float64)
    distances = np.abs(angles[:, None] - angles[None, :])
    seed_index = int(np.argmax(np.sum(distances <= radius_deg, axis=1)))
    seed = float(angles[seed_index])
    first_mask = np.abs(angles - seed) <= radius_deg
    refined_center = float(np.median(angles[first_mask]))
    return np.abs(angles - refined_center) <= radius_deg


def feature_from_samples(
    samples: list[FrameSample],
    indices: np.ndarray,
    side: str,
    minimum_frames: int,
) -> tuple[np.ndarray | None, int]:
    rows: list[tuple[float, float, float]] = []

    for sample, selected in zip(samples, indices):
        if not selected:
            continue

        if side == "LEFT" and sample.left_x is not None:
            rows.append((sample.left_x, sample.left_y, sample.angle_deg))
        elif side == "RIGHT" and sample.right_x is not None:
            rows.append((sample.right_x, sample.right_y, sample.angle_deg))
        elif (
            side == "CENTER"
            and sample.left_x is not None
            and sample.right_x is not None
        ):
            rows.append(
                (
                    0.5 * (sample.left_x + sample.right_x),
                    0.5 * (sample.left_y + sample.right_y),
                    sample.angle_deg,
                )
            )

    if len(rows) < minimum_frames:
        return None, len(rows)

    return np.median(np.asarray(rows, dtype=np.float64), axis=0), len(rows)


def format_feature(feature: np.ndarray | None) -> str:
    if feature is None:
        return "unavailable"
    return f"({feature[0]:.3f}, {feature[1]:.3f}, {feature[2]:+.4f})"


def measure_window(
    pipeline,
    *,
    label: str,
    show: bool,
    frame_count: int,
    timeout_s: float,
    angle_cluster_radius_deg: float,
    minimum_cluster_frames: int,
    minimum_side_frames: int,
) -> WindowMeasurement:
    samples: list[FrameSample] = []
    start_time = time.monotonic()
    flush_camera(pipeline)

    while len(samples) < frame_count and time.monotonic() - start_time < timeout_s:
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

        left_x = left_y = right_x = right_y = None
        if feature.left is not None:
            left_x = float(feature.left.point[0])
            left_y = float(feature.left.point[1])
        if feature.right is not None:
            right_x = float(feature.right.point[0])
            right_y = float(feature.right.point[1])

        samples.append(
            FrameSample(
                angle_deg=float(feature.top_angle_deg),
                left_x=left_x,
                left_y=left_y,
                right_x=right_x,
                right_y=right_y,
            )
        )

    if len(samples) < minimum_cluster_frames:
        raise RuntimeError(
            f"{label}: TOP 검출 부족 ({len(samples)}/{minimum_cluster_frames})"
        )

    angles = [sample.angle_deg for sample in samples]
    angle_median, _ = median_mad(angles)
    raw_angle_std = float(np.std(np.asarray(angles, dtype=np.float64)))
    cluster_mask = dominant_angle_mask(samples, angle_cluster_radius_deg)
    cluster_angles = [
        sample.angle_deg
        for sample, selected in zip(samples, cluster_mask)
        if selected
    ]

    if len(cluster_angles) < minimum_cluster_frames:
        raise RuntimeError(
            f"{label}: 지배 angle 군집 부족 "
            f"({len(cluster_angles)}/{minimum_cluster_frames}), "
            f"raw median={angle_median:+.3f} deg"
        )

    _, cluster_angle_mad = median_mad(cluster_angles)
    left, left_frames = feature_from_samples(
        samples, cluster_mask, "LEFT", minimum_side_frames
    )
    right, right_frames = feature_from_samples(
        samples, cluster_mask, "RIGHT", minimum_side_frames
    )
    center, center_frames = feature_from_samples(
        samples, cluster_mask, "CENTER", minimum_side_frames
    )

    measurement = WindowMeasurement(
        label=label,
        total_frames=len(samples),
        cluster_frames=len(cluster_angles),
        raw_angle_std_deg=raw_angle_std,
        angle_mad_deg=cluster_angle_mad,
        left_frames=left_frames,
        right_frames=right_frames,
        center_frames=center_frames,
        left=left,
        right=right,
        center=center,
    )

    print()
    print("=" * 88)
    print(
        f"[{label}] TOP={measurement.total_frames}, "
        f"dominant={measurement.cluster_frames}, "
        f"raw_std={measurement.raw_angle_std_deg:.4f} deg, "
        f"cluster_MAD={measurement.angle_mad_deg:.4f} deg"
    )
    print(f"LEFT   {format_feature(left)} | frames={left_frames}")
    print(f"RIGHT  {format_feature(right)} | frames={right_frames}")
    print(f"CENTER {format_feature(center)} | frames={center_frames}")
    print("=" * 88)
    return measurement


def measurement_to_dict(measurement: WindowMeasurement) -> dict:
    def vector(value):
        return None if value is None else [float(item) for item in value]

    return {
        "label": measurement.label,
        "total_frames": measurement.total_frames,
        "cluster_frames": measurement.cluster_frames,
        "raw_angle_std_deg": measurement.raw_angle_std_deg,
        "angle_mad_deg": measurement.angle_mad_deg,
        "left_frames": measurement.left_frames,
        "right_frames": measurement.right_frames,
        "center_frames": measurement.center_frames,
        "left": vector(measurement.left),
        "right": vector(measurement.right),
        "center": vector(measurement.center),
    }


def fit_local_model(records: list[dict], key: str) -> dict | None:
    """Fit feature = target + J @ [x(m), y(m), yaw(deg)]."""
    poses: list[np.ndarray] = []
    features: list[np.ndarray] = []
    reference_features: list[np.ndarray] = []

    for record in records:
        value = record["measurement"].get(key)
        if value is None:
            continue
        feature = np.asarray(value, dtype=np.float64)
        poses.append(np.asarray(record["actual_delta"], dtype=np.float64))
        features.append(feature)
        if record["label"].startswith("REFERENCE"):
            reference_features.append(feature)

    if len(poses) < 4 or not reference_features:
        return None

    pose_matrix = np.asarray(poses, dtype=np.float64)
    feature_matrix = np.asarray(features, dtype=np.float64)
    design = np.column_stack([np.ones(len(pose_matrix)), pose_matrix])
    coefficients, _, rank, _ = np.linalg.lstsq(design, feature_matrix, rcond=None)
    if rank < 4:
        return None

    fitted = design @ coefficients
    rmse = np.sqrt(np.mean((fitted - feature_matrix) ** 2, axis=0))
    jacobian = coefficients[1:, :].T
    target = np.median(np.asarray(reference_features), axis=0)

    return {
        "target": target.tolist(),
        "jacobian": jacobian.tolist(),
        "pose_from_feature": np.linalg.pinv(jacobian).tolist(),
        "rmse": rmse.tolist(),
        "samples": len(poses),
        "condition_number": float(np.linalg.cond(jacobian)),
    }


def print_matrix(name: str, matrix) -> None:
    print(f"{name} = np.asarray(")
    print("    [")
    for row in matrix:
        print("        [" + ", ".join(f"{value:+.6f}" for value in row) + "],")
    print("    ],")
    print("    dtype=np.float64,")
    print(")")


def print_result(name: str, result: dict | None) -> None:
    print()
    print(f"[{name}]")
    if result is None:
        print("유효 표본 또는 pose rank가 부족해 계산하지 못했습니다.")
        return

    target = result["target"]
    print(
        f"TARGET_{name} = np.asarray("
        f"[{target[0]:.6f}, {target[1]:.6f}, {target[2]:+.6f}], "
        "dtype=np.float64)"
    )
    print_matrix(f"J_{name}", result["jacobian"])
    print_matrix(f"POSE_FROM_FEATURE_{name}", result["pose_from_feature"])
    print(
        "RMSE = ["
        + ", ".join(f"{value:.4f}" for value in result["rmse"])
        + "] | "
        f"samples={result['samples']} | condition={result['condition_number']:.2f}"
    )


def main() -> None:
    address = "192.168.30.1:50051"
    camera_serial = "250122079439"
    camera_width = 640
    camera_height = 480
    camera_fps = 30

    initial_torso = np.deg2rad([0.0, 30.0, -50.0, 30.0, 0.0, 0.0]).tolist()
    head_down = np.deg2rad([-3.0, 43.0]).tolist()
    before_right = np.deg2rad(
        [-38.23, -53.19, -21.31, -48.14, -63.73, 81.18, 2.39]
    ).tolist()
    before_left = np.deg2rad(
        [-38.23, 53.19, 21.31, -48.14, 63.73, 81.18, -2.39]
    ).tolist()

    parser = argparse.ArgumentParser(description="RB-Y1 Tote local 3x3 calibration v4")
    parser.add_argument("--address", default=address)
    parser.add_argument("--model", choices=("a", "m"), default="m")
    parser.add_argument("--camera-serial", default=camera_serial)
    parser.add_argument("--show-tote", action="store_true")
    parser.add_argument("--x-step", type=float, default=0.03, help="X perturbation [m]")
    parser.add_argument("--y-step", type=float, default=0.03, help="Y perturbation [m]")
    parser.add_argument("--yaw-step", type=float, default=2.0, help="Yaw perturbation [deg]")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--angle-cluster-radius", type=float, default=0.45)
    parser.add_argument("--min-cluster-frames", type=int, default=30)
    parser.add_argument("--min-side-frames", type=int, default=20)
    parser.add_argument("--output", default="tote_local_calibration_result.json")
    args = parser.parse_args()

    robot = initialize_mobile(
        args.address,
        args.model,
        power=".*",
        servo=".*",
        unlimited=False,
    )
    monitor = OdometryMonitor()
    camera = RealSenseCamera(
        camera_width,
        camera_height,
        camera_fps,
        serial=args.camera_serial,
    )
    state_update_started = False
    camera_started = False
    records: list[dict] = []

    def measure(label: str, reference_pose) -> dict:
        current_pose = odom_pose(monitor.odom)
        actual_delta = pose_delta(reference_pose, current_pose)
        result = measure_window(
            camera.pipeline,
            label=label,
            show=args.show_tote,
            frame_count=args.frames,
            timeout_s=args.timeout,
            angle_cluster_radius_deg=args.angle_cluster_radius,
            minimum_cluster_frames=args.min_cluster_frames,
            minimum_side_frames=args.min_side_frames,
        )
        record = {
            "label": label,
            "odom_pose": [float(value) for value in current_pose],
            "actual_delta": [float(value) for value in actual_delta],
            "measurement": measurement_to_dict(result),
        }
        records.append(record)
        print(
            f"actual delta: x={actual_delta[0] * 100:+.3f} cm, "
            f"y={actual_delta[1] * 100:+.3f} cm, yaw={actual_delta[2]:+.4f} deg"
        )
        return record

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
                minimum_time=2.0,
            )
            arms_future = executor.submit(
                move_both_arms,
                robot,
                before_right,
                before_left,
                minimum_time=2.0,
            )
            torso_head_ok = torso_head_future.result()
            arms_ok = arms_future.result()

        if not torso_head_ok or not arms_ok:
            raise RuntimeError("초기 Torso / Head / BEFORE 자세 이동 실패")

        camera.start()
        camera_started = True

        print()
        print("=" * 88)
        print("로봇을 최종 파지 목표 자세에 놓아주세요.")
        print("앞으로 1 cm가 최종 목표라면 지금 그 1 cm를 포함한 자세여야 합니다.")
        print("Tote, Head/Torso, 양팔 BEFORE 자세가 실제 데모와 같은지 확인하세요.")
        print("주변 사람과 장애물을 치운 뒤 Enter를 누르면 자동 이동을 시작합니다.")
        print("=" * 88)
        input("준비되면 Enter > ")

        reference_pose = odom_pose(monitor.odom)
        measure("REFERENCE_INITIAL", reference_pose)

        perturbations = [
            ("PLUS_X", np.asarray([+args.x_step, 0.0, 0.0])),
            ("MINUS_X", np.asarray([-args.x_step, 0.0, 0.0])),
            ("PLUS_Y", np.asarray([0.0, +args.y_step, 0.0])),
            ("MINUS_Y", np.asarray([0.0, -args.y_step, 0.0])),
            ("PLUS_YAW", np.asarray([0.0, 0.0, +args.yaw_step])),
            ("MINUS_YAW", np.asarray([0.0, 0.0, -args.yaw_step])),
        ]

        for label, requested_offset in perturbations:
            print()
            print("#" * 88)
            print(
                f"{label}: requested x={requested_offset[0] * 100:+.1f} cm, "
                f"y={requested_offset[1] * 100:+.1f} cm, "
                f"yaw={requested_offset[2]:+.2f} deg"
            )
            target_pose = offset_world_pose(reference_pose, requested_offset)
            if not move_to_pose(robot, monitor, target_pose, f"{label} 이동"):
                raise RuntimeError(f"{label} 이동 실패")
            measure(label, reference_pose)

            if not move_to_pose(robot, monitor, reference_pose, f"{label} 기준 복귀"):
                raise RuntimeError(f"{label} 기준 복귀 실패")
            measure(f"REFERENCE_AFTER_{label}", reference_pose)

        results = {
            "LEFT": fit_local_model(records, "left"),
            "RIGHT": fit_local_model(records, "right"),
            "CENTER": fit_local_model(records, "center"),
        }

        print()
        print("=" * 100)
        print("TOTE LOCAL 3x3 CALIBRATION RESULT")
        print("row    = [feature_x(px), feature_y(px), top_angle(deg)]")
        print("column = [robot_x(m), robot_y(m), robot_yaw(deg)]")
        for name, result in results.items():
            print_result(name, result)
        print("=" * 100)

        output = {
            "settings": {
                "camera_serial": args.camera_serial,
                "camera_size": [camera_width, camera_height],
                "camera_fps": camera_fps,
                "x_step_m": args.x_step,
                "y_step_m": args.y_step,
                "yaw_step_deg": args.yaw_step,
                "frames": args.frames,
                "angle_cluster_radius_deg": args.angle_cluster_radius,
                "minimum_cluster_frames": args.min_cluster_frames,
                "minimum_side_frames": args.min_side_frames,
            },
            "reference_odom": [float(value) for value in reference_pose],
            "records": records,
            "results": results,
        }
        output_path = Path(args.output).expanduser()
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"결과 저장: {output_path.resolve()}")

    except KeyboardInterrupt:
        print("\n사용자가 calibration을 중단했습니다. 로봇은 현재 자세에 정지합니다.")

    finally:
        if camera_started:
            camera.stop()
        if state_update_started:
            robot.stop_state_update()
        try:
            robot.disable_control_manager()
        except Exception:
            pass
        robot.disconnect()
        if args.show_tote:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        print("모든 연결과 제어를 정리했습니다.")


if __name__ == "__main__":
    main()
