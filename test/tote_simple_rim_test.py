#!/usr/bin/env python3
"""D435 RGB에서 토트의 TOP / LEFT / RIGHT rim을 검출하고 영상 feature를 측정한다.

핵심
- Depth는 사용하지 않는다.
- TOP / LEFT / RIGHT 세 직선을 찾는다.
- TL = TOP ∩ LEFT, TR = TOP ∩ RIGHT 교점을 계산한다.
- center_x / center_y는 Hough endpoint가 아니라 TL/TR에서 계산한다.
- angle_deg는 TOP rim의 수평 기준 각도다.
- 로봇은 움직이지 않는다.

키
- p: 최근 측정값 통계 출력
- r: 측정값 초기화
- s: 현재 화면 저장
- q / ESC: 종료

사용법:
python test/tote_feature_test.py --serial 250122079439
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
# 후보선 영역 / 각도
# ============================================================

# TOP은 화면 위쪽 절반에서 찾는다.
TOP_MAX_Y_RATIO = 0.50
TOP_MAX_ANGLE_DEG = 12.0

# LEFT / RIGHT는 토트 측면의 대각선 방향을 이용한다.
LEFT_MIN_ANGLE_DEG = 90.0
LEFT_MAX_ANGLE_DEG = 145.0

RIGHT_MIN_ANGLE_DEG = 35.0
RIGHT_MAX_ANGLE_DEG = 90.0

MAX_TOP_CANDIDATES = 12
MAX_LEFT_CANDIDATES = 8
MAX_RIGHT_CANDIDATES = 8


# ============================================================
# TOP + LEFT + RIGHT 조합 검사
# ============================================================

MIN_TOP_WIDTH_RATIO = 0.45
MAX_TOP_WIDTH_RATIO = 1.10
CORNER_MARGIN_RATIO = 0.06

# TOP 아래쪽으로 이만큼 내려간 위치에서 좌우 측면이 벌어지는지 검사한다.
SIDE_PROBE_HEIGHT_RATIO = 0.30

# 검출된 세 직선 위에 실제 Canny edge가 어느 정도 존재해야 하는지 검사한다.
EDGE_SAMPLE_COUNT = 60
EDGE_SEARCH_RADIUS = 4

MIN_TOP_EDGE_SUPPORT = 0.55
MIN_SIDE_EDGE_SUPPORT = 0.35

# TOP 아래 영역이 실제 검은 토트 내부인지 확인하는 보조 조건이다.
DARK_THRESHOLD = 110
MIN_DARK_FRACTION = 0.45


# ============================================================
# 출력 / 통계
# ============================================================

HISTORY_SIZE = 60
PRINT_EVERY_N_FRAMES = 5


@dataclass
class LineCandidate:
    """Hough 선분 하나를 무한 직선으로 표현한다."""

    segment: np.ndarray
    length: float
    angle_deg: float
    center_x: float
    center_y: float
    line_abc: np.ndarray


@dataclass
class ToteFeature:
    """TOP / LEFT / RIGHT 교점에서 계산한 토트 영상 feature."""

    angle_deg: float
    center_x_px: float
    center_y_px: float
    tl: np.ndarray
    tr: np.ndarray
    left_probe: np.ndarray
    right_probe: np.ndarray
    score: float


def normalize_horizontal_angle_deg(angle_deg: float) -> float:
    """line angle을 수평 기준 -90~90 deg 범위로 변환한다."""
    angle = (float(angle_deg) + 180.0) % 180.0

    if angle >= 90.0:
        angle -= 180.0

    return angle


def line_info(segment: np.ndarray):
    """Hough 선분의 길이, 0~180도 각도, 중심점을 계산한다."""
    x1, y1, x2, y2 = [float(value) for value in segment]

    dx = x2 - x1
    dy = y2 - y1

    length = math.hypot(dx, dy)
    angle_deg = math.degrees(math.atan2(dy, dx)) % 180.0
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    return length, angle_deg, center_x, center_y


def segment_to_line(segment: np.ndarray) -> np.ndarray:
    """두 점을 ax + by + c = 0 형태의 정규화된 무한 직선으로 변환한다."""
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
    """ax + by + c = 0 직선에서 주어진 y 위치의 x를 계산한다."""
    a, b, c = line_abc

    if abs(a) < 1e-8:
        return None

    x = -(b * float(y) + c) / a

    if not np.isfinite(x):
        return None

    return float(x)


def make_candidate(segment: np.ndarray) -> LineCandidate:
    """Hough 선분을 LineCandidate로 변환한다."""
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
    """0도 또는 180도 근처의 수평선을 TOP 후보로 사용한다."""
    signed = normalize_horizontal_angle_deg(angle_deg)
    return abs(signed) <= TOP_MAX_ANGLE_DEG


def detect_line_candidates(image: np.ndarray):
    """Canny + HoughLinesP로 TOP / LEFT / RIGHT 후보선을 찾는다."""
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
        return [], [], [], gray, edges

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

        if LEFT_MIN_ANGLE_DEG <= candidate.angle_deg <= LEFT_MAX_ANGLE_DEG and candidate.center_x < width * 0.55:
            left_candidates.append(candidate)
            continue

        if RIGHT_MIN_ANGLE_DEG <= candidate.angle_deg <= RIGHT_MAX_ANGLE_DEG and candidate.center_x > width * 0.45:
            right_candidates.append(candidate)

    top_candidates.sort(key=lambda item: item.length, reverse=True)
    left_candidates.sort(key=lambda item: item.length, reverse=True)
    right_candidates.sort(key=lambda item: item.length, reverse=True)

    return (
        top_candidates[:MAX_TOP_CANDIDATES],
        left_candidates[:MAX_LEFT_CANDIDATES],
        right_candidates[:MAX_RIGHT_CANDIDATES],
        gray,
        edges,
    )


def point_inside_margin(point: np.ndarray, width: int, height: int) -> bool:
    """교점이 화면에서 지나치게 멀리 벗어나지 않는지 확인한다."""
    margin_x = width * CORNER_MARGIN_RATIO
    margin_y = height * CORNER_MARGIN_RATIO

    return bool(
        -margin_x <= point[0] <= width + margin_x
        and -margin_y <= point[1] <= height + margin_y
    )


def segment_edge_support(edges: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """두 점 사이의 예상 직선 주변에 실제 Canny edge가 얼마나 존재하는지 계산한다."""
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


def dark_fraction(gray: np.ndarray, tl: np.ndarray, tr: np.ndarray, left_probe: np.ndarray, right_probe: np.ndarray) -> float:
    """TOP 아래쪽의 사다리꼴 영역이 실제 검은 토트 내부인지 확인한다."""
    polygon = np.rint(np.stack((tl, tr, right_probe, left_probe), axis=0)).astype(np.int32)

    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)

    kernel = np.ones((11, 11), dtype=np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    pixels = gray[mask > 0]

    if pixels.size == 0:
        return 0.0

    return float(np.mean(pixels < DARK_THRESHOLD))


def evaluate_triplet(
    top: LineCandidate,
    left: LineCandidate,
    right: LineCandidate,
    gray: np.ndarray,
    edges: np.ndarray,
) -> ToteFeature | None:
    """TOP / LEFT / RIGHT 세 후보가 실제 토트 opening의 위쪽 구조인지 평가한다."""
    height, width = gray.shape

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

    # TOP에서 아래로 내려가면 LEFT는 더 왼쪽으로, RIGHT는 더 오른쪽으로 벌어져야 한다.
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

    inside_dark = dark_fraction(gray, tl, tr, left_probe, right_probe)

    if inside_dark < MIN_DARK_FRACTION:
        return None

    side_support = 0.5 * (left_support + right_support)
    length_score = min(1.0, top_width / (width * 0.80))
    score = 0.35 * top_support + 0.25 * side_support + 0.25 * inside_dark + 0.15 * length_score

    return ToteFeature(
        angle_deg=normalize_horizontal_angle_deg(top.angle_deg),
        center_x_px=float(center[0]),
        center_y_px=float(center[1]),
        tl=tl,
        tr=tr,
        left_probe=left_probe,
        right_probe=right_probe,
        score=score,
    )


def detect_tote_feature(image: np.ndarray) -> tuple[ToteFeature | None, np.ndarray]:
    """TOP / LEFT / RIGHT 후보 조합 중 가장 좋은 토트 feature를 선택한다."""
    top_candidates, left_candidates, right_candidates, gray, edges = detect_line_candidates(image)

    best_feature = None

    for top in top_candidates:
        for left in left_candidates:
            for right in right_candidates:
                feature = evaluate_triplet(top, left, right, gray, edges)

                if feature is None:
                    continue

                if best_feature is None or feature.score > best_feature.score:
                    best_feature = feature

    return best_feature, edges


def draw_feature(image: np.ndarray, feature: ToteFeature | None) -> np.ndarray:
    """TOP / LEFT / RIGHT와 TL/TR, 중심점을 영상에 표시한다."""
    output = image.copy()
    height, width = output.shape[:2]

    if feature is None:
        cv2.putText(output, "TOTE FEATURE NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return output

    tl = tuple(np.rint(feature.tl).astype(int))
    tr = tuple(np.rint(feature.tr).astype(int))
    left_probe = tuple(np.rint(feature.left_probe).astype(int))
    right_probe = tuple(np.rint(feature.right_probe).astype(int))

    center = (int(round(feature.center_x_px)), int(round(feature.center_y_px)))

    # TOP / LEFT / RIGHT 세 직선만 표시한다.
    cv2.line(output, tl, tr, (0, 255, 255), 4, cv2.LINE_AA)
    cv2.line(output, tl, left_probe, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.line(output, tr, right_probe, (255, 255, 0), 4, cv2.LINE_AA)

    cv2.circle(output, tl, 7, (0, 0, 255), -1)
    cv2.circle(output, tr, 7, (0, 0, 255), -1)
    cv2.circle(output, center, 8, (255, 0, 255), -1)

    cv2.putText(output, "TL", (tl[0] + 8, tl[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(output, "TR", (tr[0] + 8, tr[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    # 영상 정중앙은 reference 확인용으로만 표시한다.
    cv2.drawMarker(output, (width // 2, height // 2), (255, 255, 255), cv2.MARKER_CROSS, 22, 2)

    cv2.putText(output, "TOTE FEATURE FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    feature_text = (
        f"angle={feature.angle_deg:+.2f} deg  cx={feature.center_x_px:.1f}  "
        f"cy={feature.center_y_px:.1f}  score={feature.score:.3f}"
    )

    cv2.putText(output, feature_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA)

    return output


def print_feature(feature: ToteFeature) -> None:
    """현재 feature를 터미널에 출력한다."""
    print(
        f"angle={feature.angle_deg:+7.3f} deg | center_x={feature.center_x_px:7.2f} px | "
        f"center_y={feature.center_y_px:7.2f} px | score={feature.score:.3f}"
    )


def print_summary(history: deque[ToteFeature]) -> None:
    """최근 유효 측정값의 평균, 표준편차, 최소값, 최대값을 출력한다."""
    if not history:
        print("측정값이 없습니다.")
        return

    values = np.asarray(
        [[item.angle_deg, item.center_x_px, item.center_y_px] for item in history],
        dtype=np.float64,
    )

    names = ("angle_deg", "center_x_px", "center_y_px")
    units = ("deg", "px", "px")

    print()
    print("=" * 76)
    print(f"최근 유효 측정 {len(history)} frames")

    for index, (name, unit) in enumerate(zip(names, units)):
        mean = float(np.mean(values[:, index]))
        std = float(np.std(values[:, index]))
        minimum = float(np.min(values[:, index]))
        maximum = float(np.max(values[:, index]))

        print(f"{name:12s}: mean={mean:+9.3f} {unit} | std={std:7.3f} | min={minimum:+9.3f} | max={maximum:+9.3f}")

    print("=" * 76)
    print()


def start_camera(serial: str | None):
    """D435 RGB 스트림을 시작한다."""
    pipeline = rs.pipeline()
    config = rs.config()

    if serial:
        config.enable_device(serial)

    config.enable_stream(rs.stream.color, CAM_WIDTH, CAM_HEIGHT, rs.format.bgr8, CAM_FPS)
    profile = pipeline.start(config)

    for _ in range(30):
        pipeline.wait_for_frames()

    device = profile.get_device()

    try:
        camera_serial = device.get_info(rs.camera_info.serial_number)
    except Exception:
        camera_serial = "unknown"

    print(f"D435 RGB 시작: serial={camera_serial}")
    print(f"resolution={CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS}")

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=None, help="사용할 D435 serial")
    args = parser.parse_args()

    pipeline = start_camera(args.serial)
    history: deque[ToteFeature] = deque(maxlen=HISTORY_SIZE)

    frame_count = 0
    last_result = None

    print()
    print("p: 최근 측정 통계")
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
            feature, edges = detect_tote_feature(image)

            if feature is not None:
                history.append(feature)

                if frame_count % PRINT_EVERY_N_FRAMES == 0:
                    print_feature(feature)

            result = draw_feature(image, feature)
            last_result = result

            cv2.imshow("Tote 3-Line Feature", result)
            cv2.imshow("Edges", edges)

            key = cv2.waitKey(1) & 0xFF
            frame_count += 1

            if key in (27, ord("q"), ord("Q")):
                break

            if key in (ord("p"), ord("P")):
                print_summary(history)

            if key in (ord("r"), ord("R")):
                history.clear()
                print("측정값 초기화")

            if key in (ord("s"), ord("S")) and last_result is not None:
                cv2.imwrite("tote_feature.png", last_result)
                cv2.imwrite("tote_edges.png", edges)
                print("tote_feature.png / tote_edges.png 저장")

    finally:
        pipeline.stop()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()