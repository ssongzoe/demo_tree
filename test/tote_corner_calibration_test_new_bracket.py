#!/usr/bin/env python3
"""새 카메라 브라켓용 tote 검출기의 TOP/TL/TR 안정성을 확인한다.

키
- p: 최근 측정값과 처리 시간 통계 출력
- r: 측정값과 검출 추적 상태 초기화
- s: 현재 검출 화면과 edge 화면 저장
- q / ESC: 종료

권장 위치
- demo_tree/test/tote_corner_calibration_test_new_bracket.py
- demo_tree/utils/tote_vision_new_bracket.py

실행
python test/tote_corner_calibration_test_new_bracket.py --serial 250122079439
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "test" else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

from utils.tote_vision_new_bracket import (  # noqa: E402
    CAM_FPS,
    CAM_HEIGHT,
    CAM_WIDTH,
    FrameFeature,
    detect_frame_feature_with_edges,
    draw_feature,
    reset_top_tracking,
)


CAMERA_WARMUP_FRAMES = 30
WINDOW_NAME = "New Bracket Tote Detection"
EDGES_WINDOW_NAME = "New Bracket Edges"
MAX_HISTORY = 300
TIMING_HISTORY_SIZE = 300
PRINT_TIMING_EVERY_N_FRAMES = 30


def print_array_stats(name: str, values: list[float], unit: str) -> None:
    if not values:
        print(f"{name:14s}: no data")
        return

    array = np.asarray(values, dtype=np.float64)
    print(
        f"{name:14s}: mean={np.mean(array):+9.3f} | "
        f"median={np.median(array):+9.3f} | std={np.std(array):7.3f} | "
        f"min={np.min(array):+9.3f} | max={np.max(array):+9.3f} {unit}"
    )


def print_timing_stats(detect_times_ms, frame_wait_times_ms) -> None:
    if not detect_times_ms:
        print("시간 측정값이 없습니다.")
        return

    detect = np.asarray(detect_times_ms, dtype=np.float64)
    wait = np.asarray(frame_wait_times_ms, dtype=np.float64)

    print()
    print("=" * 96)
    print(f"최근 {len(detect)} frame 시간 통계")
    print(
        f"Detector    : mean={np.mean(detect):7.2f} ms | "
        f"median={np.median(detect):7.2f} ms | "
        f"P90={np.percentile(detect, 90):7.2f} ms | max={np.max(detect):7.2f} ms"
    )

    if wait.size:
        print(
            f"Camera wait : mean={np.mean(wait):7.2f} ms | "
            f"median={np.median(wait):7.2f} ms | "
            f"P90={np.percentile(wait, 90):7.2f} ms | max={np.max(wait):7.2f} ms"
        )

    print(f"Detector FPS equivalent: {1000.0 / np.mean(detect):.1f} FPS")
    print("=" * 96)
    print()


def print_stats(history: list[FrameFeature]) -> None:
    if not history:
        print("측정값이 없습니다.")
        return

    left = [feature for feature in history if feature.left is not None]
    right = [feature for feature in history if feature.right is not None]
    both = [
        feature
        for feature in history
        if feature.left is not None and feature.right is not None
    ]

    print()
    print("=" * 96)
    print(
        f"최근 측정 {len(history)} frames | TOP={len(history)} | "
        f"LEFT={len(left)} | RIGHT={len(right)} | BOTH={len(both)}"
    )
    print("-" * 96)
    print_array_stats(
        "top_angle",
        [feature.top_angle_deg for feature in history],
        "deg",
    )

    print()
    print("[LEFT feature = TL.x, TL.y, top_angle]")
    print_array_stats("TL.x", [feature.left.point[0] for feature in left], "px")
    print_array_stats("TL.y", [feature.left.point[1] for feature in left], "px")
    print_array_stats(
        "LEFT angle",
        [feature.top_angle_deg for feature in left],
        "deg",
    )

    print()
    print("[RIGHT feature = TR.x, TR.y, top_angle]")
    print_array_stats("TR.x", [feature.right.point[0] for feature in right], "px")
    print_array_stats("TR.y", [feature.right.point[1] for feature in right], "px")
    print_array_stats(
        "RIGHT angle",
        [feature.top_angle_deg for feature in right],
        "deg",
    )

    if both:
        widths = [
            float(np.linalg.norm(feature.right.point - feature.left.point))
            for feature in both
        ]
        print()
        print_array_stats("TOP width", widths, "px")

    print("=" * 96)
    print()


def start_camera(serial: str | None) -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()

    if serial:
        config.enable_device(serial)

    config.enable_stream(
        rs.stream.color,
        CAM_WIDTH,
        CAM_HEIGHT,
        rs.format.bgr8,
        CAM_FPS,
    )
    pipeline.start(config)

    for _ in range(CAMERA_WARMUP_FRAMES):
        pipeline.wait_for_frames()

    print(f"D435 시작: {CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS}")
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=None, help="사용할 D435 serial")
    parser.add_argument(
        "--save-dir",
        default="new_bracket_vision_test",
        help="s 키로 저장할 이미지 폴더",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    pipeline = start_camera(args.serial)
    history: list[FrameFeature] = []
    detect_times_ms = deque(maxlen=TIMING_HISTORY_SIZE)
    frame_wait_times_ms = deque(maxlen=TIMING_HISTORY_SIZE)
    frame_index = 0
    last_output = None
    last_edges = None

    reset_top_tracking()

    print()
    print("새 브라켓용 TOP + TL/TR 검출 테스트")
    print("p: 측정값 + detector 시간 통계")
    print("r: 측정값 + 검출 추적 상태 초기화")
    print("s: 현재 검출 화면과 edge 화면 저장")
    print("q / ESC: 종료")
    print()

    try:
        while True:
            wait_start = time.perf_counter()
            frames = pipeline.wait_for_frames()
            frame_wait_times_ms.append(
                (time.perf_counter() - wait_start) * 1000.0
            )

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asarray(color_frame.get_data())

            detect_start = time.perf_counter()
            feature, edges = detect_frame_feature_with_edges(image)
            detect_times_ms.append(
                (time.perf_counter() - detect_start) * 1000.0
            )

            output = draw_feature(image, feature, label="NEW BRACKET")
            frame_index += 1

            if frame_index % PRINT_TIMING_EVERY_N_FRAMES == 0:
                print(
                    f"Detection timing | current={detect_times_ms[-1]:.2f} ms | "
                    f"mean={np.mean(detect_times_ms):.2f} ms | "
                    f"median={np.median(detect_times_ms):.2f} ms"
                )

            if feature is not None:
                history.append(feature)
                history = history[-MAX_HISTORY:]

            last_output = output
            last_edges = edges

            cv2.imshow(WINDOW_NAME, output)
            cv2.imshow(EDGES_WINDOW_NAME, edges)
            cv2.moveWindow(WINDOW_NAME, 0, 0)
            cv2.moveWindow(EDGES_WINDOW_NAME, CAM_WIDTH + 10, 0)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("r"):
                history.clear()
                detect_times_ms.clear()
                frame_wait_times_ms.clear()
                reset_top_tracking()
                print("측정값 / 시간 통계 / 검출 추적 상태 초기화")
            elif key == ord("p"):
                print_stats(history)
                print_timing_stats(detect_times_ms, frame_wait_times_ms)
            elif key == ord("s") and last_output is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                result_path = save_dir / f"new_bracket_tote_detection_{timestamp}.png"
                edges_path = save_dir / f"new_bracket_tote_edges_{timestamp}.png"

                cv2.imwrite(str(result_path), last_output)
                if last_edges is not None:
                    cv2.imwrite(str(edges_path), last_edges)

                print(f"저장: {result_path}")
                if last_edges is not None:
                    print(f"저장: {edges_path}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
