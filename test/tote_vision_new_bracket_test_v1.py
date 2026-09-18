#!/usr/bin/env python3
"""New head-bracket tote vision diagnostic.

This script never moves the robot.  It opens the head camera, temporarily
widens the tote detector limits, and displays:

    left  : RGB + the feature selected by utils/tote_vision.py
    right : Canny edges + raw TOP/SIDE Hough candidates

Keys:
    p : print rolling statistics
    s : save the current screen and statistics JSON
    r : clear rolling statistics
    q / ESC : quit
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import sys

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "control").is_dir() else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.mobile_controller import initialize_mobile  # noqa: E402
from control.robot_controller import move_both_arms, move_torso_and_head  # noqa: E402
from utils import tote_vision as vision  # noqa: E402


WINDOW_NAME = "Tote Vision - New Bracket Test v1"
ROLLING_FRAMES = 120


def segment_points(segment: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    values = np.rint(segment).astype(int)
    return (int(values[0]), int(values[1])), (int(values[2]), int(values[3]))


def draw_horizontal_grid(image: np.ndarray) -> None:
    """Draw image y coordinates so the new TOP ROI can be read directly."""
    height, width = image.shape[:2]

    for y in range(0, height, 25):
        color = (80, 80, 80) if y % 50 else (130, 130, 130)
        cv2.line(image, (0, y), (width - 1, y), color, 1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"y={y}",
            (5, min(height - 5, y + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_raw_top_candidates(edges: np.ndarray, output: np.ndarray) -> int:
    """Draw every near-horizontal Hough line inside the diagnostic TOP ROI."""
    roi_edges = np.zeros_like(edges)
    roi_edges[vision.TOP_MIN_Y:vision.TOP_MAX_Y, :] = edges[
        vision.TOP_MIN_Y:vision.TOP_MAX_Y, :
    ]
    detected = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=vision.HOUGH_THRESHOLD,
        minLineLength=vision.TOP_FRAGMENT_MIN_LINE_LENGTH,
        maxLineGap=vision.MAX_LINE_GAP,
    )

    count = 0
    if detected is None:
        return count

    for segment in np.asarray(detected, dtype=np.int32).reshape(-1, 4):
        candidate = vision.make_candidate(segment)
        angle = vision.normalize_angle(candidate.angle_deg)

        if abs(angle) > vision.TOP_MAX_ANGLE_DEG:
            continue

        p1, p2 = segment_points(candidate.segment)
        cv2.line(output, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)
        count += 1

    return count


def draw_side_candidates(edges: np.ndarray, output: np.ndarray) -> tuple[int, int]:
    """Draw every SIDE candidate used by the current detector."""
    left_candidates, right_candidates = vision.detect_side_candidates(edges)

    for candidate in left_candidates:
        p1, p2 = segment_points(candidate.segment)
        cv2.line(output, p1, p2, (0, 220, 0), 2, cv2.LINE_AA)

    for candidate in right_candidates:
        p1, p2 = segment_points(candidate.segment)
        cv2.line(output, p1, p2, (255, 180, 0), 2, cv2.LINE_AA)

    return len(left_candidates), len(right_candidates)


def empty_history() -> dict[str, deque[float]]:
    return {
        "top_y": deque(maxlen=ROLLING_FRAMES),
        "angle": deque(maxlen=ROLLING_FRAMES),
        "left_x": deque(maxlen=ROLLING_FRAMES),
        "left_y": deque(maxlen=ROLLING_FRAMES),
        "right_x": deque(maxlen=ROLLING_FRAMES),
        "right_y": deque(maxlen=ROLLING_FRAMES),
        "width": deque(maxlen=ROLLING_FRAMES),
    }


def update_history(history: dict[str, deque[float]], feature) -> None:
    if feature is None:
        return

    top_y = vision.line_y_at_x(feature.top.segment, vision.CAM_WIDTH * 0.5)
    if top_y is not None:
        history["top_y"].append(float(top_y))
    history["angle"].append(float(feature.top_angle_deg))

    if feature.left is not None:
        history["left_x"].append(float(feature.left.point[0]))
        history["left_y"].append(float(feature.left.point[1]))

    if feature.right is not None:
        history["right_x"].append(float(feature.right.point[0]))
        history["right_y"].append(float(feature.right.point[1]))

    if feature.left is not None and feature.right is not None:
        width = float(np.linalg.norm(feature.right.point - feature.left.point))
        history["width"].append(width)


def summarize(values: deque[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}

    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def statistics(history: dict[str, deque[float]]) -> dict[str, dict]:
    return {name: summarize(values) for name, values in history.items()}


def print_statistics(history: dict[str, deque[float]]) -> None:
    print()
    print("=" * 88)
    print(f"ROLLING DETECTION STATISTICS (maximum {ROLLING_FRAMES} frames)")
    for name, result in statistics(history).items():
        if result["count"] == 0:
            print(f"{name:>8}: unavailable")
            continue
        print(
            f"{name:>8}: n={result['count']:3d}, "
            f"mean={result['mean']:+8.3f}, std={result['std']:7.3f}, "
            f"range=[{result['min']:+8.3f}, {result['max']:+8.3f}]"
        )
    print("=" * 88)


def current_status_text(feature) -> str:
    if feature is None:
        return "TOP=--  TL=--  TR=--  width=--"

    left_text = "--"
    right_text = "--"
    width_text = "--"

    if feature.left is not None:
        left_text = f"({feature.left.point[0]:.1f},{feature.left.point[1]:.1f})"
    if feature.right is not None:
        right_text = f"({feature.right.point[0]:.1f},{feature.right.point[1]:.1f})"
    if feature.left is not None and feature.right is not None:
        width = np.linalg.norm(feature.right.point - feature.left.point)
        width_text = f"{width:.1f}px"

    return (
        f"angle={feature.top_angle_deg:+.2f}  "
        f"TL={left_text}  TR={right_text}  width={width_text}"
    )


def save_result(
    output_dir: Path,
    combined: np.ndarray,
    history: dict[str, deque[float]],
    args,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = output_dir / f"new_bracket_tote_detection_{stamp}.png"
    json_path = output_dir / f"new_bracket_tote_detection_{stamp}.json"
    cv2.imwrite(str(image_path), combined)

    payload = {
        "camera_serial": args.camera_serial,
        "top_min_y": vision.TOP_MIN_Y,
        "top_max_y": vision.TOP_MAX_Y,
        "left_corner_max_x_ratio": vision.LEFT_CORNER_MAX_X_RATIO,
        "right_corner_min_x_ratio": vision.RIGHT_CORNER_MIN_X_RATIO,
        "min_top_width_ratio": vision.MIN_TOP_WIDTH_RATIO,
        "max_top_width_ratio": vision.MAX_TOP_WIDTH_RATIO,
        "corner_margin_ratio": vision.CORNER_MARGIN_RATIO,
        "statistics": statistics(history),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"화면 저장: {image_path.resolve()}")
    print(f"통계 저장: {json_path.resolve()}")


def apply_diagnostic_limits(args) -> None:
    """Widen only this process; utils/tote_vision.py is not modified."""
    vision.TOP_MIN_Y = args.top_min_y
    vision.TOP_MAX_Y = args.top_max_y
    vision.LEFT_CORNER_MAX_X_RATIO = args.left_max_x_ratio
    vision.RIGHT_CORNER_MIN_X_RATIO = args.right_min_x_ratio
    vision.MIN_TOP_WIDTH_RATIO = args.min_width_ratio
    vision.MAX_TOP_WIDTH_RATIO = args.max_width_ratio
    vision.CORNER_MARGIN_RATIO = args.corner_margin_ratio


def main() -> None:
    address = "192.168.30.1:50051"
    camera_serial = "250122079439"
    initial_torso = np.deg2rad([0.0, 30.0, -50.0, 30.0, 0.0, 0.0]).tolist()
    head_down = np.deg2rad([0.0, 43.0]).tolist()
    before_right = np.deg2rad(
        [-38.23, -53.19, -21.31, -48.14, -63.73, 81.18, 2.39]
    ).tolist()
    before_left = np.deg2rad(
        [-38.23, 53.19, 21.31, -48.14, 63.73, 81.18, -2.39]
    ).tolist()

    parser = argparse.ArgumentParser(
        description="Stationary tote detection test for the new camera bracket"
    )
    parser.add_argument("--address", default=address)
    parser.add_argument("--model", choices=("a", "m"), default="m")
    parser.add_argument("--camera-serial", default=camera_serial)
    parser.add_argument("--skip-init-pose", action="store_true")
    parser.add_argument("--top-min-y", type=int, default=20)
    parser.add_argument("--top-max-y", type=int, default=280)
    parser.add_argument("--left-max-x-ratio", type=float, default=0.50)
    parser.add_argument("--right-min-x-ratio", type=float, default=0.50)
    parser.add_argument("--min-width-ratio", type=float, default=0.40)
    parser.add_argument("--max-width-ratio", type=float, default=1.30)
    parser.add_argument("--corner-margin-ratio", type=float, default=0.15)
    parser.add_argument("--display-scale", type=float, default=0.75)
    parser.add_argument("--output-dir", default="new_bracket_vision_test")
    args = parser.parse_args()

    if not 0 <= args.top_min_y < args.top_max_y <= vision.CAM_HEIGHT:
        raise ValueError("TOP y range must satisfy 0 <= min < max <= image height")

    apply_diagnostic_limits(args)
    history = empty_history()
    robot = None
    pipeline = None

    try:
        if not args.skip_init_pose:
            robot = initialize_mobile(
                args.address,
                args.model,
                power=".*",
                servo=".*",
                unlimited=False,
            )
            print("기존 데모와 같은 Torso / Head / 양팔 BEFORE 자세로 이동")
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

        pipeline = vision.start_camera(args.camera_serial)
        print("베이스는 움직이지 않습니다.")
        print("p=통계, s=화면/통계 저장, r=통계 초기화, q/ESC=종료")
        print(
            f"진단 범위: TOP y={vision.TOP_MIN_Y}..{vision.TOP_MAX_Y}, "
            f"width={vision.MIN_TOP_WIDTH_RATIO:.2f}..{vision.MAX_TOP_WIDTH_RATIO:.2f}W"
        )

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asarray(color_frame.get_data())
            gray, edges = vision.preprocess(image)
            feature = vision.detect_frame_feature(image)
            update_history(history, feature)

            rgb_view = vision.draw_feature(image, feature, label="NEW BRACKET")
            cv2.line(
                rgb_view,
                (0, vision.TOP_MIN_Y),
                (vision.CAM_WIDTH - 1, vision.TOP_MIN_Y),
                (255, 0, 255),
                2,
            )
            cv2.line(
                rgb_view,
                (0, vision.TOP_MAX_Y),
                (vision.CAM_WIDTH - 1, vision.TOP_MAX_Y),
                (255, 0, 255),
                2,
            )
            cv2.putText(
                rgb_view,
                current_status_text(feature),
                (15, vision.CAM_HEIGHT - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            edge_view = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            draw_horizontal_grid(edge_view)
            top_count = draw_raw_top_candidates(edges, edge_view)
            left_count, right_count = draw_side_candidates(edges, edge_view)
            cv2.line(
                edge_view,
                (0, vision.TOP_MIN_Y),
                (vision.CAM_WIDTH - 1, vision.TOP_MIN_Y),
                (255, 0, 255),
                2,
            )
            cv2.line(
                edge_view,
                (0, vision.TOP_MAX_Y),
                (vision.CAM_WIDTH - 1, vision.TOP_MAX_Y),
                (255, 0, 255),
                2,
            )
            cv2.putText(
                edge_view,
                f"raw TOP={top_count}  LEFT={left_count}  RIGHT={right_count}",
                (15, vision.CAM_HEIGHT - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            combined = cv2.hconcat([rgb_view, edge_view])
            display = cv2.resize(
                combined,
                None,
                fx=args.display_scale,
                fy=args.display_scale,
                interpolation=cv2.INTER_AREA,
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                print_statistics(history)
            elif key == ord("s"):
                print_statistics(history)
                save_result(Path(args.output_dir), combined, history, args)
            elif key == ord("r"):
                history = empty_history()
                if hasattr(vision, "reset_top_tracking"):
                    vision.reset_top_tracking()
                print("통계를 초기화했습니다.")
    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()
        if robot is not None:
            try:
                robot.disable_control_manager()
            except Exception:
                pass
            robot.disconnect()


if __name__ == "__main__":
    main()
