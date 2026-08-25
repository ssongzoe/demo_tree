#!/usr/bin/env python3
"""Yaw 정렬 후 토트의 FINE feature 기준값을 측정한다.

TOP + LEFT + RIGHT 세 직선을 이용해 TL/TR 교점을 계산하고 center_x / center_y를 출력한다.

키
- p: 최근 측정 통계 출력
- r: 측정값 초기화
- s: 현재 화면 저장
- q / ESC: 종료

사용법:
python test/tote_fine_measure_test.py --serial 250122079439
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np
import pyrealsense2 as rs


# ============================================================
# 카메라
# ============================================================

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30


# ============================================================
# Edge / Hough
# ============================================================

CANNY_LOW = 40
CANNY_HIGH = 120

HOUGH_THRESHOLD = 45
MIN_LINE_LENGTH = 60
MAX_LINE_GAP = 40


# ============================================================
# 후보선
# ============================================================

TOP_MAX_Y_RATIO = 0.50
TOP_MAX_ANGLE_DEG = 12.0

LEFT_MIN_ANGLE_DEG = 90.0
LEFT_MAX_ANGLE_DEG = 145.0

RIGHT_MIN_ANGLE_DEG = 35.0
RIGHT_MAX_ANGLE_DEG = 90.0

MAX_TOP_CANDIDATES = 6
MAX_LEFT_CANDIDATES = 4
MAX_RIGHT_CANDIDATES = 4


# ============================================================
# FINE 검출 조건
# ============================================================

MIN_TOP_WIDTH_RATIO = 0.45
MAX_TOP_WIDTH_RATIO = 1.10

CORNER_MARGIN_RATIO = 0.06
SIDE_PROBE_HEIGHT_RATIO = 0.30

EDGE_SAMPLE_COUNT = 30
EDGE_SEARCH_RADIUS = 4

MIN_TOP_EDGE_SUPPORT = 0.55
MIN_SIDE_EDGE_SUPPORT = 0.35


# ============================================================
# 측정
# ============================================================

HISTORY_SIZE = 60
PRINT_EVERY_N_FRAMES = 5


@dataclass
class LineCandidate:
    """Hough 후보 직선."""

    segment: np.ndarray
    length: float
    angle_deg: float
    center_x: float
    center_y: float
    line_abc: np.ndarray


@dataclass
class FineFeature:
    """TL/TR에서 계산한 정밀 토트 feature."""

    angle_deg: float
    center_x_px: float
    center_y_px: float
    top_width_px: float

    tl: np.ndarray
    tr: np.ndarray
    left_probe: np.ndarray
    right_probe: np.ndarray

    score: float


def normalize_horizontal_angle_deg(angle_deg: float) -> float:
    """직선 각도를 수평 기준 -90~90도로 변환한다."""
    angle = (float(angle_deg) + 180.0) % 180.0

    if angle >= 90.0:
        angle -= 180.0

    return angle


def line_info(segment: np.ndarray):
    """Hough 선분의 길이, 각도, 중심을 계산한다."""
    x1, y1, x2, y2 = [float(value) for value in segment]

    dx = x2 - x1
    dy = y2 - y1

    length = math.hypot(dx, dy)
    angle_deg = math.degrees(math.atan2(dy, dx)) % 180.0
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    return length, angle_deg, center_x, center_y


def segment_to_line(segment: np.ndarray) -> np.ndarray:
    """선분을 ax + by + c = 0 형태의 무한 직선으로 변환한다."""
    x1, y1, x2, y2 = [float(value) for value in segment]

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    norm = math.hypot(a, b)

    if norm < 1e-9:
        raise ValueError("길이가 0인 선분입니다.")

    return np.asarray([a / norm, b / norm, c / norm], dtype=np.float64)


def line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    """두 무한 직선의 교점을 계산한다."""
    a1, b1, c1 = first
    a2, b2, c2 = second

    determinant = a1 * b2 - a2 * b1

    if abs(determinant) < 1e-8:
        return None

    x = (b1 * c2 - b2 * c1) / determinant
    y = (c1 * a2 - c2 * a1) / determinant

    if not np.isfinite(x) or not np.isfinite(y):
        return None

    return np.asarray([x, y], dtype=np.float64)


def line_x_at_y(line_abc: np.ndarray, y: float) -> float | None:
    """무한 직선에서 지정 y 위치의 x를 계산한다."""
    a, b, c = line_abc

    if abs(a) < 1e-8:
        return None

    x = -(b * float(y) + c) / a

    if not np.isfinite(x):
        return None

    return float(x)


def make_candidate(segment: np.ndarray) -> LineCandidate:
    """Hough 선분을 후보 객체로 만든다."""
    length, angle_deg, center_x, center_y = line_info(segment)

    return LineCandidate(
        segment=np.asarray(segment, dtype=np.float64),
        length=length,
        angle_deg=angle_deg,
        center_x=center_x,
        center_y=center_y,
        line_abc=segment_to_line(segment),
    )


def is_top_angle(angle_deg: float) -> bool:
    """수평에 가까운 선인지 확인한다."""
    return abs(normalize_horizontal_angle_deg(angle_deg)) <= TOP_MAX_ANGLE_DEG


def detect_line_candidates(image: np.ndarray):
    """TOP / LEFT / RIGHT Hough 후보를 찾는다."""
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    detected = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    if detected is None:
        return [], [], [], edges

    segments = np.asarray(detected, dtype=np.int32).reshape(-1, 4)

    top_candidates = []
    left_candidates = []
    right_candidates = []

    for segment in segments:
        candidate = make_candidate(segment)

        if candidate.length < MIN_LINE_LENGTH:
            continue

        if is_top_angle(candidate.angle_deg) and candidate.center_y < height * TOP_MAX_Y_RATIO:
            top_candidates.append(candidate)
            continue

        if LEFT_MIN_ANGLE_DEG <= candidate.angle_deg <= LEFT_MAX_ANGLE_DEG and candidate.center_x < width * 0.65:
            left_candidates.append(candidate)
            continue

        if RIGHT_MIN_ANGLE_DEG <= candidate.angle_deg <= RIGHT_MAX_ANGLE_DEG and candidate.center_x > width * 0.35:
            right_candidates.append(candidate)

    top_candidates.sort(key=lambda item: item.length, reverse=True)
    left_candidates.sort(key=lambda item: item.length, reverse=True)
    right_candidates.sort(key=lambda item: item.length, reverse=True)

    return (
        top_candidates[:MAX_TOP_CANDIDATES],
        left_candidates[:MAX_LEFT_CANDIDATES],
        right_candidates[:MAX_RIGHT_CANDIDATES],
        edges,
    )


def point_inside_margin(point: np.ndarray, width: int, height: int) -> bool:
    """교점이 영상 주변의 허용 범위 안에 있는지 확인한다."""
    margin_x = width * CORNER_MARGIN_RATIO
    margin_y = height * CORNER_MARGIN_RATIO

    return bool(-margin_x <= point[0] <= width + margin_x and -margin_y <= point[1] <= height + margin_y)


def segment_edge_support(edges: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """예상 직선 위에 실제 Canny edge가 얼마나 존재하는지 계산한다."""
    height, width = edges.shape

    xs = np.linspace(start[0], end[0], EDGE_SAMPLE_COUNT)
    ys = np.linspace(start[1], end[1], EDGE_SAMPLE_COUNT)

    supported = 0
    valid = 0

    for x_value, y_value in zip(xs, ys):
        x = int(round(x_value))
        y = int(round(y_value))

        if not (0 <= x < width and 0 <= y < height):
            continue

        x0 = max(0, x - EDGE_SEARCH_RADIUS)
        x1 = min(width, x + EDGE_SEARCH_RADIUS + 1)
        y0 = max(0, y - EDGE_SEARCH_RADIUS)
        y1 = min(height, y + EDGE_SEARCH_RADIUS + 1)

        valid += 1

        if np.any(edges[y0:y1, x0:x1] > 0):
            supported += 1

    if valid == 0:
        return 0.0

    return supported / valid


def evaluate_triplet(
    top: LineCandidate,
    left: LineCandidate,
    right: LineCandidate,
    edges: np.ndarray,
) -> FineFeature | None:
    """TOP / LEFT / RIGHT 조합으로 TL/TR과 중심점을 계산한다."""
    height, width = edges.shape

    tl = line_intersection(top.line_abc, left.line_abc)
    tr = line_intersection(top.line_abc, right.line_abc)

    if tl is None or tr is None:
        return None

    if not point_inside_margin(tl, width, height) or not point_inside_margin(tr, width, height):
        return None

    if tl[0] >= tr[0]:
        return None

    top_width = float(np.linalg.norm(tr - tl))

    if top_width < width * MIN_TOP_WIDTH_RATIO or top_width > width * MAX_TOP_WIDTH_RATIO:
        return None

    center = 0.5 * (tl + tr)

    if center[1] < height * 0.05 or center[1] > height * TOP_MAX_Y_RATIO:
        return None

    probe_y = min(height * 0.90, center[1] + height * SIDE_PROBE_HEIGHT_RATIO)

    left_probe_x = line_x_at_y(left.line_abc, probe_y)
    right_probe_x = line_x_at_y(right.line_abc, probe_y)

    if left_probe_x is None or right_probe_x is None:
        return None

    left_probe = np.asarray([left_probe_x, probe_y], dtype=np.float64)
    right_probe = np.asarray([right_probe_x, probe_y], dtype=np.float64)

    if left_probe[0] >= tl[0] or right_probe[0] <= tr[0]:
        return None

    if left_probe[0] >= right_probe[0]:
        return None

    top_support = segment_edge_support(edges, tl, tr)
    left_support = segment_edge_support(edges, tl, left_probe)
    right_support = segment_edge_support(edges, tr, right_probe)

    if top_support < MIN_TOP_EDGE_SUPPORT:
        return None

    if left_support < MIN_SIDE_EDGE_SUPPORT or right_support < MIN_SIDE_EDGE_SUPPORT:
        return None

    side_support = 0.5 * (left_support + right_support)
    width_score = min(1.0, top_width / (width * 0.80))

    score = 0.50 * top_support + 0.30 * side_support + 0.20 * width_score

    return FineFeature(
        angle_deg=normalize_horizontal_angle_deg(top.angle_deg),
        center_x_px=float(center[0]),
        center_y_px=float(center[1]),
        top_width_px=top_width,
        tl=tl,
        tr=tr,
        left_probe=left_probe,
        right_probe=right_probe,
        score=score,
    )


def detect_fine_feature(image: np.ndarray) -> tuple[FineFeature | None, np.ndarray]:
    """모든 후보 조합 중 가장 좋은 FINE feature를 선택한다."""
    top_candidates, left_candidates, right_candidates, edges = detect_line_candidates(image)

    best_feature = None

    for top in top_candidates:
        for left in left_candidates:
            for right in right_candidates:
                feature = evaluate_triplet(top, left, right, edges)

                if feature is None:
                    continue

                if best_feature is None or feature.score > best_feature.score:
                    best_feature = feature

    return best_feature, edges


def draw_feature(image: np.ndarray, feature: FineFeature | None) -> np.ndarray:
    """FINE 검출 결과를 화면에 표시한다."""
    output = image.copy()

    if feature is None:
        cv2.putText(output, "FINE NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        return output

    tl = tuple(np.rint(feature.tl).astype(int))
    tr = tuple(np.rint(feature.tr).astype(int))
    left_probe = tuple(np.rint(feature.left_probe).astype(int))
    right_probe = tuple(np.rint(feature.right_probe).astype(int))

    center = (int(round(feature.center_x_px)), int(round(feature.center_y_px)))

    cv2.line(output, tl, tr, (0, 255, 255), 4, cv2.LINE_AA)
    cv2.line(output, tl, left_probe, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.line(output, tr, right_probe, (255, 255, 0), 4, cv2.LINE_AA)

    cv2.circle(output, tl, 7, (0, 0, 255), -1)
    cv2.circle(output, tr, 7, (0, 0, 255), -1)
    cv2.circle(output, center, 8, (255, 0, 255), -1)

    text = (
        f"angle={feature.angle_deg:+.2f}  cx={feature.center_x_px:.1f}  "
        f"cy={feature.center_y_px:.1f}  width={feature.top_width_px:.1f}"
    )

    cv2.putText(output, "FINE FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(output, text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA)

    return output


def print_feature(feature: FineFeature) -> None:
    """현재 FINE feature를 출력한다."""
    print(
        f"angle={feature.angle_deg:+7.3f} deg | center_x={feature.center_x_px:7.2f} px | "
        f"center_y={feature.center_y_px:7.2f} px | width={feature.top_width_px:7.2f} px | score={feature.score:.3f}"
    )


def print_summary(history: deque[FineFeature]) -> None:
    """최근 FINE 측정값의 통계를 출력한다."""
    if not history:
        print("FINE 측정값이 없습니다.")
        return

    values = np.asarray(
        [[item.angle_deg, item.center_x_px, item.center_y_px, item.top_width_px] for item in history],
        dtype=np.float64,
    )

    names = ("angle_deg", "center_x_px", "center_y_px", "top_width_px")
    units = ("deg", "px", "px", "px")

    print()
    print("=" * 82)
    print(f"최근 FINE 측정 {len(history)} frames")

    for index, (name, unit) in enumerate(zip(names, units)):
        mean = float(np.mean(values[:, index]))
        median = float(np.median(values[:, index]))
        std = float(np.std(values[:, index]))
        minimum = float(np.min(values[:, index]))
        maximum = float(np.max(values[:, index]))

        print(
            f"{name:12s}: mean={mean:+9.3f} | median={median:+9.3f} | std={std:7.3f} | "
            f"min={minimum:+9.3f} | max={maximum:+9.3f} {unit}"
        )

    print("=" * 82)
    print()


def start_camera(serial: str | None) -> rs.pipeline:
    """D435 RGB 스트림을 시작한다."""
    pipeline = rs.pipeline()
    config = rs.config()

    if serial:
        config.enable_device(serial)

    config.enable_stream(rs.stream.color, CAM_WIDTH, CAM_HEIGHT, rs.format.bgr8, CAM_FPS)
    pipeline.start(config)

    for _ in range(30):
        pipeline.wait_for_frames()

    print(f"D435 시작: {CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS}")

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=None, help="사용할 D435 serial")
    args = parser.parse_args()

    pipeline = start_camera(args.serial)
    history: deque[FineFeature] = deque(maxlen=HISTORY_SIZE)

    frame_count = 0
    last_result = None
    last_edges = None

    print()
    print("Yaw 정렬된 기준 자세에서 측정하세요.")
    print("p: 최근 60 frame 통계")
    print("r: 측정값 초기화")
    print("s: 현재 화면 저장")
    print("q / ESC: 종료")
    print()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            image = np.asarray(color_frame.get_data())
            feature, edges = detect_fine_feature(image)

            if feature is not None:
                history.append(feature)

                if frame_count % PRINT_EVERY_N_FRAMES == 0:
                    print_feature(feature)

            result = draw_feature(image, feature)

            last_result = result
            last_edges = edges

            cv2.imshow("Tote Fine Measure", result)
            cv2.imshow("Edges", edges)

            key = cv2.waitKey(1) & 0xFF
            frame_count += 1

            if key in (27, ord("q"), ord("Q")):
                break

            if key in (ord("r"), ord("R")):
                history.clear()
                print("측정값 초기화")

            if key in (ord("p"), ord("P")):
                print_summary(history)

            if key in (ord("s"), ord("S")) and last_result is not None:
                cv2.imwrite("tote_fine.png", last_result)

                if last_edges is not None:
                    cv2.imwrite("tote_fine_edges.png", last_edges)

                print("tote_fine.png / tote_fine_edges.png 저장")

    finally:
        pipeline.stop()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()