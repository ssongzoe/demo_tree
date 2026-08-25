#!/usr/bin/env python3
"""D435 RGB에서 토트의 TOP / LEFT / RIGHT rim을 검출한다.

FINE
- TOP + LEFT + RIGHT가 모두 검출되면 TL/TR 교점을 사용한다.
- center_x / center_y는 TL/TR의 중심으로 계산한다.

COARSE
- LEFT 또는 RIGHT가 화면 밖으로 나가 FINE 검출이 실패하면 TOP rim만 사용한다.
- TOP rim의 대략적인 중심을 이용해 토트가 화면 왼쪽/오른쪽 어느 방향에 있는지 판단한다.

현재 버전에서는 로봇을 움직이지 않는다.

키
- p: 최근 측정 통계
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

EDGE_SAMPLE_COUNT = 40
EDGE_SEARCH_RADIUS = 4

MIN_TOP_EDGE_SUPPORT = 0.55
MIN_SIDE_EDGE_SUPPORT = 0.35

DARK_THRESHOLD = 110
MIN_DARK_FRACTION = 0.45


# ============================================================
# COARSE 검출 조건
# ============================================================

# 가장 긴 TOP 후보 주변에 있는 비슷한 Hough line들을 하나의 rim으로 묶는다.
COARSE_TOP_Y_TOL_PX = 12.0
COARSE_TOP_ANGLE_TOL_DEG = 3.0

# 나중에 로봇 복귀 제어에 사용할 안전 영역이다. 현재는 화면에만 표시한다.
COARSE_CENTER_LEFT_PX = 280.0
COARSE_CENTER_RIGHT_PX = 380.0


# ============================================================
# 출력
# ============================================================

HISTORY_SIZE = 60
PRINT_EVERY_N_FRAMES = 5


@dataclass
class LineCandidate:
    """Hough 선분 하나와 무한 직선 표현."""

    segment: np.ndarray
    length: float
    angle_deg: float
    center_x: float
    center_y: float
    line_abc: np.ndarray


@dataclass
class ToteFeature:
    """토트 영상 feature."""

    mode: str
    angle_deg: float
    center_x_px: float
    center_y_px: float
    score: float

    top_p1: np.ndarray
    top_p2: np.ndarray

    tl: np.ndarray | None = None
    tr: np.ndarray | None = None
    left_probe: np.ndarray | None = None
    right_probe: np.ndarray | None = None


def normalize_horizontal_angle_deg(angle_deg: float) -> float:
    """line angle을 수평 기준 -90~90도 범위로 변환한다."""
    angle = (float(angle_deg) + 180.0) % 180.0

    if angle >= 90.0:
        angle -= 180.0

    return angle


def angle_difference_deg(first: float, second: float) -> float:
    """수평선 방향 두 개의 최소 각도 차이를 계산한다."""
    return abs(normalize_horizontal_angle_deg(first - second))


def line_info(segment: np.ndarray):
    """Hough 선분의 길이, 각도, 중심점을 계산한다."""
    x1, y1, x2, y2 = [float(value) for value in segment]

    dx = x2 - x1
    dy = y2 - y1

    length = math.hypot(dx, dy)
    angle_deg = math.degrees(math.atan2(dy, dx)) % 180.0
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    return length, angle_deg, center_x, center_y


def segment_to_line(segment: np.ndarray) -> np.ndarray:
    """두 점을 ax + by + c = 0 형태의 무한 직선으로 변환한다."""
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
    """ax + by + c = 0 직선에서 지정 y의 x를 계산한다."""
    a, b, c = line_abc

    if abs(a) < 1e-8:
        return None

    x = -(b * float(y) + c) / a

    if not np.isfinite(x):
        return None

    return float(x)


def fitted_y_at_x(vx: float, vy: float, x0: float, y0: float, x: float) -> float:
    """cv2.fitLine 직선에서 지정 x 위치의 y를 계산한다."""
    if abs(vx) < 1e-8:
        return float(y0)

    return float(y0 + (x - x0) * vy / vx)


def make_candidate(segment: np.ndarray) -> LineCandidate:
    """Hough 선분을 후보 객체로 변환한다."""
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
    """수평에 가까운 선인지 검사한다."""
    return abs(normalize_horizontal_angle_deg(angle_deg)) <= TOP_MAX_ANGLE_DEG


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
        gray,
        edges,
    )


def point_inside_margin(point: np.ndarray, width: int, height: int) -> bool:
    """교점이 화면에서 지나치게 멀리 벗어나지 않는지 확인한다."""
    margin_x = width * CORNER_MARGIN_RATIO
    margin_y = height * CORNER_MARGIN_RATIO

    return bool(-margin_x <= point[0] <= width + margin_x and -margin_y <= point[1] <= height + margin_y)


def segment_edge_support(edges: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """예상 직선 주변에 실제 Canny edge가 얼마나 존재하는지 계산한다."""
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
    """TOP 아래 사다리꼴 영역이 실제 검은 토트 내부인지 확인한다."""
    polygon = np.rint(np.stack((tl, tr, right_probe, left_probe), axis=0)).astype(np.int32)

    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)

    kernel = np.ones((11, 11), dtype=np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    pixels = gray[mask > 0]

    if pixels.size == 0:
        return 0.0

    return float(np.mean(pixels < DARK_THRESHOLD))


def evaluate_fine_triplet(
    top: LineCandidate,
    left: LineCandidate,
    right: LineCandidate,
    gray: np.ndarray,
    edges: np.ndarray,
) -> ToteFeature | None:
    """TOP / LEFT / RIGHT 세 후보를 이용해 FINE feature를 계산한다."""
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
        mode="FINE",
        angle_deg=normalize_horizontal_angle_deg(top.angle_deg),
        center_x_px=float(center[0]),
        center_y_px=float(center[1]),
        score=score,
        top_p1=tl,
        top_p2=tr,
        tl=tl,
        tr=tr,
        left_probe=left_probe,
        right_probe=right_probe,
    )


def detect_fine_feature(
    top_candidates: list[LineCandidate],
    left_candidates: list[LineCandidate],
    right_candidates: list[LineCandidate],
    gray: np.ndarray,
    edges: np.ndarray,
) -> ToteFeature | None:
    """후보 조합 중 가장 좋은 FINE 검출 결과를 선택한다."""
    best_feature = None

    for top in top_candidates:
        for left in left_candidates:
            for right in right_candidates:
                feature = evaluate_fine_triplet(top, left, right, gray, edges)

                if feature is None:
                    continue

                if best_feature is None or feature.score > best_feature.score:
                    best_feature = feature

    return best_feature


def detect_coarse_feature(top_candidates: list[LineCandidate]) -> ToteFeature | None:
    """FINE 검출이 실패하면 TOP rim만 이용해 대략적인 위치와 각도를 계산한다."""
    if not top_candidates:
        return None

    anchor = top_candidates[0]
    grouped = []

    for candidate in top_candidates:
        if abs(candidate.center_y - anchor.center_y) > COARSE_TOP_Y_TOL_PX:
            continue

        if angle_difference_deg(candidate.angle_deg, anchor.angle_deg) > COARSE_TOP_ANGLE_TOL_DEG:
            continue

        grouped.append(candidate)

    if not grouped:
        grouped = [anchor]

    points = []

    for candidate in grouped:
        x1, y1, x2, y2 = candidate.segment
        points.append((x1, y1))
        points.append((x2, y2))

    points = np.asarray(points, dtype=np.float32)

    vx, vy, fit_x, fit_y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)

    if vx < 0.0:
        vx = -vx
        vy = -vy

    angle_deg = normalize_horizontal_angle_deg(math.degrees(math.atan2(float(vy), float(vx))))

    left_x = float(np.min(points[:, 0]))
    right_x = float(np.max(points[:, 0]))

    left_y = fitted_y_at_x(float(vx), float(vy), float(fit_x), float(fit_y), left_x)
    right_y = fitted_y_at_x(float(vx), float(vy), float(fit_x), float(fit_y), right_x)

    center_x = 0.5 * (left_x + right_x)
    center_y = fitted_y_at_x(float(vx), float(vy), float(fit_x), float(fit_y), center_x)

    top_p1 = np.asarray([left_x, left_y], dtype=np.float64)
    top_p2 = np.asarray([right_x, right_y], dtype=np.float64)

    # COARSE score는 실제 정밀도 의미가 아니라 사용한 TOP 후보의 평균 길이를 정규화한 참고값이다.
    mean_length = float(np.mean([candidate.length for candidate in grouped]))
    score = min(1.0, mean_length / 400.0)

    return ToteFeature(
        mode="COARSE",
        angle_deg=angle_deg,
        center_x_px=center_x,
        center_y_px=center_y,
        score=score,
        top_p1=top_p1,
        top_p2=top_p2,
    )


def detect_tote_feature(image: np.ndarray) -> tuple[ToteFeature | None, np.ndarray]:
    """FINE 검출을 먼저 시도하고 실패하면 TOP-only COARSE 검출로 전환한다."""
    top_candidates, left_candidates, right_candidates, gray, edges = detect_line_candidates(image)

    fine_feature = detect_fine_feature(top_candidates, left_candidates, right_candidates, gray, edges)

    if fine_feature is not None:
        return fine_feature, edges

    coarse_feature = detect_coarse_feature(top_candidates)

    return coarse_feature, edges


def draw_feature(image: np.ndarray, feature: ToteFeature | None) -> np.ndarray:
    """FINE 또는 COARSE 검출 결과를 영상에 표시한다."""
    output = image.copy()
    height, width = output.shape[:2]

    # COARSE에서 화면 안쪽으로 돌아왔다고 판단할 영역을 표시한다.
    cv2.line(output, (int(COARSE_CENTER_LEFT_PX), 0), (int(COARSE_CENTER_LEFT_PX), height), (100, 100, 100), 1)
    cv2.line(output, (int(COARSE_CENTER_RIGHT_PX), 0), (int(COARSE_CENTER_RIGHT_PX), height), (100, 100, 100), 1)

    cv2.drawMarker(output, (width // 2, height // 2), (255, 255, 255), cv2.MARKER_CROSS, 22, 2)

    if feature is None:
        cv2.putText(output, "TOTE NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        return output

    top_p1 = tuple(np.rint(feature.top_p1).astype(int))
    top_p2 = tuple(np.rint(feature.top_p2).astype(int))
    center = (int(round(feature.center_x_px)), int(round(feature.center_y_px)))

    if feature.mode == "FINE":
        color = (0, 255, 0)

        tl = tuple(np.rint(feature.tl).astype(int))
        tr = tuple(np.rint(feature.tr).astype(int))
        left_probe = tuple(np.rint(feature.left_probe).astype(int))
        right_probe = tuple(np.rint(feature.right_probe).astype(int))

        cv2.line(output, tl, tr, (0, 255, 255), 4, cv2.LINE_AA)
        cv2.line(output, tl, left_probe, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.line(output, tr, right_probe, (255, 255, 0), 4, cv2.LINE_AA)

        cv2.circle(output, tl, 7, (0, 0, 255), -1)
        cv2.circle(output, tr, 7, (0, 0, 255), -1)

        cv2.putText(output, "TL", (tl[0] + 8, tl[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(output, "TR", (tr[0] + 8, tr[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    else:
        color = (0, 165, 255)
        cv2.line(output, top_p1, top_p2, color, 4, cv2.LINE_AA)

    cv2.circle(output, center, 8, (255, 0, 255), -1)

    title = f"{feature.mode} MODE"
    cv2.putText(output, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    text = (
        f"angle={feature.angle_deg:+.2f} deg  cx={feature.center_x_px:.1f}  "
        f"cy={feature.center_y_px:.1f}  score={feature.score:.3f}"
    )

    cv2.putText(output, text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

    if feature.mode == "COARSE":
        if feature.center_x_px < COARSE_CENTER_LEFT_PX:
            direction_text = "TOTE LEFT  -> base should move RIGHT"
        elif feature.center_x_px > COARSE_CENTER_RIGHT_PX:
            direction_text = "TOTE RIGHT -> base should move LEFT"
        else:
            direction_text = "TOTE INSIDE RECOVERY AREA"

        cv2.putText(output, direction_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return output


def print_feature(feature: ToteFeature) -> None:
    """현재 feature를 터미널에 출력한다."""
    print(
        f"{feature.mode:6s} | angle={feature.angle_deg:+7.3f} deg | center_x={feature.center_x_px:7.2f} px | "
        f"center_y={feature.center_y_px:7.2f} px | score={feature.score:.3f}"
    )


def print_stats(title: str, features: list[ToteFeature]) -> None:
    """지정 mode의 측정 통계를 출력한다."""
    if not features:
        print(f"{title}: 측정값 없음")
        return

    values = np.asarray([[item.angle_deg, item.center_x_px, item.center_y_px] for item in features], dtype=np.float64)

    names = ("angle_deg", "center_x_px", "center_y_px")
    units = ("deg", "px", "px")

    print()
    print(f"[{title}] {len(features)} frames")

    for index, (name, unit) in enumerate(zip(names, units)):
        mean = float(np.mean(values[:, index]))
        std = float(np.std(values[:, index]))
        minimum = float(np.min(values[:, index]))
        maximum = float(np.max(values[:, index]))

        print(f"{name:12s}: mean={mean:+9.3f} {unit} | std={std:7.3f} | min={minimum:+9.3f} | max={maximum:+9.3f}")


def print_summary(history: deque[ToteFeature]) -> None:
    """FINE과 COARSE 측정값을 분리해서 출력한다."""
    if not history:
        print("측정값이 없습니다.")
        return

    fine_features = [item for item in history if item.mode == "FINE"]
    coarse_features = [item for item in history if item.mode == "COARSE"]

    print()
    print("=" * 82)
    print(f"최근 유효 측정 {len(history)} frames | FINE={len(fine_features)} | COARSE={len(coarse_features)}")

    print_stats("FINE", fine_features)
    print_stats("COARSE", coarse_features)

    print()
    print("=" * 82)
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
    last_edges = None

    print()
    print("FINE   : TOP + LEFT + RIGHT")
    print("COARSE : TOP only")
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
            last_edges = edges

            cv2.imshow("Tote Feature", result)
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

                if last_edges is not None:
                    cv2.imwrite("tote_edges.png", last_edges)

                print("tote_feature.png / tote_edges.png 저장")

    finally:
        pipeline.stop()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()