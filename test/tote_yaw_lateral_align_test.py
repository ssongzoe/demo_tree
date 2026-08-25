#!/usr/bin/env python3
"""토트 TOP rim으로 yaw를 한 번 정렬한 뒤 FINE center_x로 lateral을 한 번 정렬한다.

동작
1. TOP rim 여러 frame 측정
2. yaw 오차 계산
3. base yaw 한 번 이동
4. yaw 결과 검증
5. TOP + LEFT + RIGHT로 FINE center_x 측정
6. lateral 오차 계산
7. base lateral 한 번 이동
8. lateral 결과 검증

추가 보정은 하지 않는다.

사용법:
python test/tote_yaw_lateral_align_test.py --serial 250122079439
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import math
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs


# ============================================================
# 프로젝트 import 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.mobile_controller import OdometryMonitor, build_leg, initialize_mobile, move_leg, odom_pose, wait_for_odometry


# ============================================================
# 로봇
# ============================================================

ADDRESS = "192.168.30.1:50051"

SETTLE_S = 0.7

QUINTIC_PEAK = 1.875
MIN_LEG_TIME = 1.5

ALIGN_ANGULAR_SPEED = 0.5
ALIGN_LINEAR_SPEED = 0.08


# ============================================================
# 카메라
# ============================================================

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

CAMERA_WARMUP_FRAMES = 30
CAMERA_FLUSH_FRAMES = 10


# ============================================================
# Yaw 기준값
# ============================================================

TARGET_ANGLE_DEG = 0.0
ANGLE_TOL_DEG = 1.0

# 실제 실험에서 image angle -4.965도에 base +7.324도가 거의 정확하게 맞았다.
YAW_GAIN = 1.475
YAW_SIGN = -1.0

MAX_YAW_COMMAND_DEG = 10.0


# ============================================================
# Lateral 기준값
# ============================================================

# 실제 grasp 기준 위치에서 측정한 FINE median.
TARGET_CENTER_X_PX = 339.36

# 기준 위치의 center_y도 이후 forward 정렬에 사용할 예정이다.
TARGET_CENTER_Y_PX = 103.00

CENTER_X_TOL_PX = 5.0

# LEFT/RIGHT ±2 cm 실험에서 약 11.1 px/cm가 측정되었다.
LATERAL_M_PER_PX = 0.00090

# +Y=LEFT 기준. current cx가 클수록 base는 RIGHT(-Y)로 이동한다.
LATERAL_SIGN = 1.0

MAX_LATERAL_COMMAND_M = 0.04


# ============================================================
# 공통 Edge / Hough
# ============================================================

CANNY_LOW = 40
CANNY_HIGH = 120

HOUGH_THRESHOLD = 45
MIN_LINE_LENGTH = 60
MAX_LINE_GAP = 40


# ============================================================
# TOP-only Yaw 검출
# ============================================================

TOP_ONLY_MIN_LINE_LENGTH = 100
TOP_ONLY_MAX_ANGLE_DEG = 12.0

TOP_ONLY_MIN_Y = 55
TOP_ONLY_MAX_Y = 175

CONTRAST_OFFSET_PX = 10
CONTRAST_SAMPLE_COUNT = 20

TOP_MEASURE_FRAMES = 20
TOP_MEASURE_TIMEOUT_S = 4.0

MAX_TOP_ANGLE_STD_DEG = 0.25
MAX_TOP_CENTER_Y_STD_PX = 2.0


# ============================================================
# FINE 검출
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

MIN_TOP_WIDTH_RATIO = 0.45
MAX_TOP_WIDTH_RATIO = 1.10

CORNER_MARGIN_RATIO = 0.06
SIDE_PROBE_HEIGHT_RATIO = 0.30

EDGE_SAMPLE_COUNT = 30
EDGE_SEARCH_RADIUS = 4

MIN_TOP_EDGE_SUPPORT = 0.55
MIN_SIDE_EDGE_SUPPORT = 0.35

FINE_MEASURE_FRAMES = 20
FINE_MEASURE_TIMEOUT_S = 5.0

FINE_WIDTH_FILTER_PX = 10.0
FINE_CENTER_Y_FILTER_PX = 5.0

FINE_TOP_MAX_ANGLE_DEG = 1.0


@dataclass
class TopFeature:
    """한 frame의 TOP rim."""

    angle_deg: float
    center_x_px: float
    center_y_px: float
    length_px: float
    contrast: float
    p1: tuple[int, int]
    p2: tuple[int, int]


@dataclass
class TopMeasurement:
    """여러 TOP frame의 안정화된 측정 결과."""

    angle_deg: float
    center_x_px: float
    center_y_px: float
    angle_std_deg: float
    center_y_std_px: float
    valid_frames: int


@dataclass
class LineCandidate:
    """FINE 검출용 Hough 직선 후보."""

    segment: np.ndarray
    length: float
    angle_deg: float
    center_x: float
    center_y: float
    line_abc: np.ndarray


@dataclass
class FineFeature:
    """TOP / LEFT / RIGHT 교점에서 계산한 FINE feature."""

    angle_deg: float
    center_x_px: float
    center_y_px: float
    top_width_px: float

    tl: np.ndarray
    tr: np.ndarray
    left_probe: np.ndarray
    right_probe: np.ndarray

    score: float


@dataclass
class FineMeasurement:
    """여러 FINE frame의 median 측정 결과."""

    angle_deg: float
    center_x_px: float
    center_y_px: float
    top_width_px: float

    center_x_std_px: float
    center_y_std_px: float

    valid_frames: int


def clamp(value: float, minimum: float, maximum: float) -> float:
    """값을 지정 범위 안으로 제한한다."""
    return max(minimum, min(maximum, value))


def normalize_horizontal_angle_deg(angle_deg: float) -> float:
    """직선 각도를 수평 기준 -90~90도로 변환한다."""
    angle = (float(angle_deg) + 180.0) % 180.0

    if angle >= 90.0:
        angle -= 180.0

    return angle


def line_info(segment: np.ndarray):
    """선분의 길이, 각도, 중심을 계산한다."""
    x1, y1, x2, y2 = [float(value) for value in segment]

    dx = x2 - x1
    dy = y2 - y1

    length = math.hypot(dx, dy)
    angle_deg = math.degrees(math.atan2(dy, dx)) % 180.0
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    return length, angle_deg, center_x, center_y


def line_y_at_x(segment: np.ndarray, x: float) -> float | None:
    """선분의 무한 직선에서 지정 x 위치의 y를 계산한다."""
    x1, y1, x2, y2 = [float(value) for value in segment]

    if abs(x2 - x1) < 1e-8:
        return None

    return y1 + (float(x) - x1) * (y2 - y1) / (x2 - x1)


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


# ============================================================
# TOP-only detector
# ============================================================


def sample_line_contrast(gray: np.ndarray, line: np.ndarray) -> float:
    """TOP 바로 위와 아래의 밝기 차이를 계산한다."""
    height, width = gray.shape

    x1, _, x2, _ = [float(value) for value in line]
    xs = np.linspace(min(x1, x2), max(x1, x2), CONTRAST_SAMPLE_COUNT)

    above_values = []
    below_values = []

    for x_value in xs:
        y_value = line_y_at_x(line, x_value)

        if y_value is None:
            continue

        x = int(round(x_value))
        y = int(round(y_value))

        y_above = y - CONTRAST_OFFSET_PX
        y_below = y + CONTRAST_OFFSET_PX

        if not (1 <= x < width - 1 and 1 <= y_above < height - 1 and 1 <= y_below < height - 1):
            continue

        above_patch = gray[y_above - 1:y_above + 2, x - 1:x + 2]
        below_patch = gray[y_below - 1:y_below + 2, x - 1:x + 2]

        above_values.append(float(np.mean(above_patch)))
        below_values.append(float(np.mean(below_patch)))

    if not above_values:
        return -1000.0

    return float(np.mean(above_values) - np.mean(below_values))


def detect_top_rim(image: np.ndarray) -> TopFeature | None:
    """Yaw 정렬에 사용할 TOP rim 하나를 검출한다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    roi_edges = np.zeros_like(edges)
    roi_edges[TOP_ONLY_MIN_Y:TOP_ONLY_MAX_Y, :] = edges[TOP_ONLY_MIN_Y:TOP_ONLY_MAX_Y, :]

    detected = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=HOUGH_THRESHOLD,
        minLineLength=TOP_ONLY_MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    if detected is None:
        return None

    lines = np.asarray(detected, dtype=np.int32).reshape(-1, 4)

    best_feature = None
    best_score = -float("inf")

    for line in lines:
        length, angle_deg_raw, center_x, center_y = line_info(line)
        angle_deg = normalize_horizontal_angle_deg(angle_deg_raw)

        if length < TOP_ONLY_MIN_LINE_LENGTH or abs(angle_deg) > TOP_ONLY_MAX_ANGLE_DEG:
            continue

        if not TOP_ONLY_MIN_Y <= center_y <= TOP_ONLY_MAX_Y:
            continue

        contrast = sample_line_contrast(gray, line)

        length_score = length / CAM_WIDTH
        contrast_score = np.clip(contrast / 80.0, -1.0, 1.0)
        score = 0.70 * length_score + 0.30 * contrast_score

        if score <= best_score:
            continue

        x1, y1, x2, y2 = [int(value) for value in line]
        p1, p2 = ((x1, y1), (x2, y2)) if x1 <= x2 else ((x2, y2), (x1, y1))

        best_score = score
        best_feature = TopFeature(angle_deg, center_x, center_y, length, contrast, p1, p2)

    return best_feature


def draw_top(image: np.ndarray, feature: TopFeature | None, title: str) -> np.ndarray:
    """TOP detector 결과를 화면에 표시한다."""
    output = image.copy()

    if feature is None:
        cv2.putText(output, "TOP NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return output

    center = (int(round(feature.center_x_px)), int(round(feature.center_y_px)))

    cv2.line(output, feature.p1, feature.p2, (0, 255, 255), 4, cv2.LINE_AA)
    cv2.circle(output, center, 7, (0, 0, 255), -1)

    text = f"{title}  angle={feature.angle_deg:+.2f}  cx={feature.center_x_px:.1f}  cy={feature.center_y_px:.1f}"
    cv2.putText(output, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    return output


def measure_top(pipeline: rs.pipeline, title: str) -> TopMeasurement | None:
    """여러 TOP frame을 측정하고 median angle을 계산한다."""
    features = []
    start_time = time.monotonic()

    while len(features) < TOP_MEASURE_FRAMES and time.monotonic() - start_time < TOP_MEASURE_TIMEOUT_S:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asarray(color_frame.get_data())
        feature = detect_top_rim(image)

        cv2.imshow("Tote Align", draw_top(image, feature, title))
        cv2.waitKey(1)

        if feature is not None:
            features.append(feature)

    if len(features) < TOP_MEASURE_FRAMES // 2:
        print(f"TOP 측정 실패: {len(features)}/{TOP_MEASURE_FRAMES}")
        return None

    center_ys = np.asarray([feature.center_y_px for feature in features], dtype=np.float64)
    median_y = float(np.median(center_ys))

    filtered = [feature for feature in features if abs(feature.center_y_px - median_y) <= 5.0]

    if len(filtered) < TOP_MEASURE_FRAMES // 2:
        print("TOP 측정 안정화 실패")
        return None

    angles = np.asarray([feature.angle_deg for feature in filtered], dtype=np.float64)
    center_xs = np.asarray([feature.center_x_px for feature in filtered], dtype=np.float64)
    center_ys = np.asarray([feature.center_y_px for feature in filtered], dtype=np.float64)

    measurement = TopMeasurement(
        angle_deg=float(np.median(angles)),
        center_x_px=float(np.median(center_xs)),
        center_y_px=float(np.median(center_ys)),
        angle_std_deg=float(np.std(angles)),
        center_y_std_px=float(np.std(center_ys)),
        valid_frames=len(filtered),
    )

    print(
        f"TOP 측정 | angle={measurement.angle_deg:+.3f} deg | cx={measurement.center_x_px:.1f} px | "
        f"cy={measurement.center_y_px:.1f} px | angle_std={measurement.angle_std_deg:.3f} | "
        f"cy_std={measurement.center_y_std_px:.3f} | frames={measurement.valid_frames}"
    )

    if measurement.angle_std_deg > MAX_TOP_ANGLE_STD_DEG or measurement.center_y_std_px > MAX_TOP_CENTER_Y_STD_PX:
        print("TOP 측정이 불안정합니다.")
        return None

    return measurement


# ============================================================
# FINE detector
# ============================================================


def make_candidate(segment: np.ndarray) -> LineCandidate:
    """Hough 선분을 FINE 후보 객체로 변환한다."""
    length, angle_deg, center_x, center_y = line_info(segment)

    return LineCandidate(
        segment=np.asarray(segment, dtype=np.float64),
        length=length,
        angle_deg=angle_deg,
        center_x=center_x,
        center_y=center_y,
        line_abc=segment_to_line(segment),
    )


def detect_fine_candidates(image: np.ndarray):
    """TOP / LEFT / RIGHT 후보선을 검출한다."""
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

    top_candidates = []
    left_candidates = []
    right_candidates = []

    for segment in np.asarray(detected, dtype=np.int32).reshape(-1, 4):
        candidate = make_candidate(segment)
        signed_angle = normalize_horizontal_angle_deg(candidate.angle_deg)

        if abs(signed_angle) <= FINE_TOP_MAX_ANGLE_DEG and candidate.center_y < height * TOP_MAX_Y_RATIO:
            top_candidates.append(candidate)
        elif LEFT_MIN_ANGLE_DEG <= candidate.angle_deg <= LEFT_MAX_ANGLE_DEG and candidate.center_x < width * 0.65:
            left_candidates.append(candidate)
        elif RIGHT_MIN_ANGLE_DEG <= candidate.angle_deg <= RIGHT_MAX_ANGLE_DEG and candidate.center_x > width * 0.35:
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
    """교점이 영상 주변 허용 범위에 있는지 확인한다."""
    margin_x = width * CORNER_MARGIN_RATIO
    margin_y = height * CORNER_MARGIN_RATIO

    return bool(-margin_x <= point[0] <= width + margin_x and -margin_y <= point[1] <= height + margin_y)


def segment_edge_support(edges: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """예상 직선 주변에 실제 edge가 얼마나 존재하는지 계산한다."""
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

    return supported / valid if valid else 0.0


def evaluate_fine_triplet(top: LineCandidate, left: LineCandidate, right: LineCandidate, edges: np.ndarray) -> FineFeature | None:
    """TOP / LEFT / RIGHT 세 직선으로 TL/TR과 FINE feature를 계산한다."""
    height, width = edges.shape

    tl = line_intersection(top.line_abc, left.line_abc)
    tr = line_intersection(top.line_abc, right.line_abc)

    if tl is None or tr is None:
        return None

    if not point_inside_margin(tl, width, height) or not point_inside_margin(tr, width, height) or tl[0] >= tr[0]:
        return None

    top_width = float(np.linalg.norm(tr - tl))

    if not width * MIN_TOP_WIDTH_RATIO <= top_width <= width * MAX_TOP_WIDTH_RATIO:
        return None

    center = 0.5 * (tl + tr)
    probe_y = min(height * 0.90, center[1] + height * SIDE_PROBE_HEIGHT_RATIO)

    left_probe_x = line_x_at_y(left.line_abc, probe_y)
    right_probe_x = line_x_at_y(right.line_abc, probe_y)

    if left_probe_x is None or right_probe_x is None:
        return None

    left_probe = np.asarray([left_probe_x, probe_y], dtype=np.float64)
    right_probe = np.asarray([right_probe_x, probe_y], dtype=np.float64)

    if left_probe[0] >= tl[0] or right_probe[0] <= tr[0] or left_probe[0] >= right_probe[0]:
        return None

    top_support = segment_edge_support(edges, tl, tr)
    left_support = segment_edge_support(edges, tl, left_probe)
    right_support = segment_edge_support(edges, tr, right_probe)

    if top_support < MIN_TOP_EDGE_SUPPORT or left_support < MIN_SIDE_EDGE_SUPPORT or right_support < MIN_SIDE_EDGE_SUPPORT:
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


def detect_fine_feature(image: np.ndarray) -> FineFeature | None:
    """유효한 조합 중 TOP 폭이 가장 큰 FINE feature를 선택한다."""
    top_candidates, left_candidates, right_candidates, edges = detect_fine_candidates(image)

    valid_features = []

    for top in top_candidates:
        for left in left_candidates:
            for right in right_candidates:
                feature = evaluate_fine_triplet(top, left, right, edges)

                if feature is not None:
                    valid_features.append(feature)

    if not valid_features:
        return None

    return max(valid_features, key=lambda feature: feature.top_width_px)


def draw_fine(image: np.ndarray, feature: FineFeature | None, title: str) -> np.ndarray:
    """FINE detector 결과를 표시한다."""
    output = image.copy()

    if feature is None:
        cv2.putText(output, "FINE NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
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

    text = f"{title}  angle={feature.angle_deg:+.2f}  cx={feature.center_x_px:.1f}  cy={feature.center_y_px:.1f}"
    cv2.putText(output, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    return output


def measure_fine(pipeline: rs.pipeline, title: str) -> FineMeasurement | None:
    """FINE feature 여러 frame을 측정하고 median 값을 계산한다."""
    features = []
    start_time = time.monotonic()

    while len(features) < FINE_MEASURE_FRAMES and time.monotonic() - start_time < FINE_MEASURE_TIMEOUT_S:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asarray(color_frame.get_data())
        feature = detect_fine_feature(image)

        cv2.imshow("Tote Align", draw_fine(image, feature, title))
        cv2.waitKey(1)

        if feature is not None:
            features.append(feature)

    if len(features) < FINE_MEASURE_FRAMES // 2:
        print(f"FINE 측정 실패: {len(features)}/{FINE_MEASURE_FRAMES}")
        return None

    center_ys = np.asarray([feature.center_y_px for feature in features], dtype=np.float64)
    widths = np.asarray([feature.top_width_px for feature in features], dtype=np.float64)

    median_center_y = float(np.median(center_ys))
    median_width = float(np.median(widths))

    filtered = [
        feature for feature in features
        if abs(feature.center_y_px - median_center_y) <= FINE_CENTER_Y_FILTER_PX
        and abs(feature.top_width_px - median_width) <= FINE_WIDTH_FILTER_PX
    ]

    if len(filtered) < FINE_MEASURE_FRAMES // 2:
        print("FINE 측정 안정화 실패")
        return None

    center_xs = np.asarray([feature.center_x_px for feature in filtered], dtype=np.float64)
    center_ys = np.asarray([feature.center_y_px for feature in filtered], dtype=np.float64)
    angles = np.asarray([feature.angle_deg for feature in filtered], dtype=np.float64)
    widths = np.asarray([feature.top_width_px for feature in filtered], dtype=np.float64)

    measurement = FineMeasurement(
        angle_deg=float(np.median(angles)),
        center_x_px=float(np.median(center_xs)),
        center_y_px=float(np.median(center_ys)),
        top_width_px=float(np.median(widths)),
        center_x_std_px=float(np.std(center_xs)),
        center_y_std_px=float(np.std(center_ys)),
        valid_frames=len(filtered),
    )

    print(
        f"FINE 측정 | angle={measurement.angle_deg:+.3f} deg | cx={measurement.center_x_px:.2f} px | "
        f"cy={measurement.center_y_px:.2f} px | width={measurement.top_width_px:.2f} px | "
        f"cx_std={measurement.center_x_std_px:.3f} | cy_std={measurement.center_y_std_px:.3f} | "
        f"frames={measurement.valid_frames}"
    )

    return measurement


# ============================================================
# Base 이동
# ============================================================


def flush_camera(pipeline: rs.pipeline) -> None:
    """로봇 이동 중 쌓인 이전 frame을 버린다."""
    for _ in range(CAMERA_FLUSH_FRAMES):
        pipeline.wait_for_frames()


def turn_duration(angle_rad: float) -> float:
    """Yaw trajectory 시간을 계산한다."""
    return max(QUINTIC_PEAK * abs(angle_rad) / ALIGN_ANGULAR_SPEED, MIN_LEG_TIME)


def linear_duration(distance_m: float) -> float:
    """Lateral trajectory 시간을 계산한다."""
    return max(QUINTIC_PEAK * abs(distance_m) / ALIGN_LINEAR_SPEED, MIN_LEG_TIME)


def move_relative_yaw(robot, monitor: OdometryMonitor, angle_deg: float) -> bool:
    """현재 자세 기준으로 yaw만 상대 이동한다."""
    angle_rad = math.radians(angle_deg)

    leg = build_leg(
        start=odom_pose(monitor.odom),
        target=(0.0, 0.0, angle_rad),
        absolute=False,
        duration=turn_duration(angle_rad),
        turn_direction="shortest",
    )

    return move_leg(robot, monitor, leg, settle=SETTLE_S)


def move_relative_lateral(robot, monitor: OdometryMonitor, lateral_y_m: float) -> bool:
    """현재 자세 기준으로 body Y 방향만 상대 이동한다."""
    leg = build_leg(
        start=odom_pose(monitor.odom),
        target=(0.0, lateral_y_m, 0.0),
        absolute=False,
        duration=linear_duration(lateral_y_m),
        turn_direction="shortest",
    )

    return move_leg(robot, monitor, leg, settle=SETTLE_S)


# ============================================================
# Align
# ============================================================


def align_yaw_once(robot, monitor: OdometryMonitor, pipeline: rs.pipeline) -> bool:
    """TOP angle을 한 번 측정하고 yaw를 한 번에 보정한다."""
    print()
    print("========== YAW ALIGN ==========")

    before = measure_top(pipeline, "YAW BEFORE")

    if before is None:
        return False

    error_deg = before.angle_deg - TARGET_ANGLE_DEG

    if abs(error_deg) <= ANGLE_TOL_DEG:
        print("이미 Yaw 정렬 범위 안입니다.")
        return True

    correction_deg = YAW_SIGN * YAW_GAIN * error_deg
    correction_deg = clamp(correction_deg, -MAX_YAW_COMMAND_DEG, MAX_YAW_COMMAND_DEG)

    print(f"Yaw error        : {error_deg:+.3f} deg")
    print(f"Base yaw command : {correction_deg:+.3f} deg")

    if not move_relative_yaw(robot, monitor, correction_deg):
        return False

    time.sleep(SETTLE_S)
    flush_camera(pipeline)

    after = measure_top(pipeline, "YAW AFTER")

    if after is None:
        return False

    final_error = after.angle_deg - TARGET_ANGLE_DEG

    print(f"Yaw before : {before.angle_deg:+.3f} deg")
    print(f"Yaw after  : {after.angle_deg:+.3f} deg")
    print(f"Yaw error  : {final_error:+.3f} deg")

    return abs(final_error) <= ANGLE_TOL_DEG


def align_lateral_once(robot, monitor: OdometryMonitor, pipeline: rs.pipeline) -> bool:
    """FINE center_x를 한 번 측정하고 lateral을 한 번에 보정한다."""
    print()
    print("========== LATERAL ALIGN ==========")

    before = measure_fine(pipeline, "LATERAL BEFORE")

    if before is None:
        return False

    error_px = before.center_x_px - TARGET_CENTER_X_PX

    if abs(error_px) <= CENTER_X_TOL_PX:
        print("이미 Lateral 정렬 범위 안입니다.")
        return True

    lateral_y_m = LATERAL_SIGN * (TARGET_CENTER_X_PX - before.center_x_px) * LATERAL_M_PER_PX
    lateral_y_m = clamp(lateral_y_m, -MAX_LATERAL_COMMAND_M, MAX_LATERAL_COMMAND_M)

    print(f"Center_x current : {before.center_x_px:.2f} px")
    print(f"Center_x target  : {TARGET_CENTER_X_PX:.2f} px")
    print(f"Center_x error   : {error_px:+.2f} px")
    print(f"Base y command   : {lateral_y_m:+.4f} m")

    if not move_relative_lateral(robot, monitor, lateral_y_m):
        return False

    time.sleep(SETTLE_S)
    flush_camera(pipeline)

    after = measure_fine(pipeline, "LATERAL AFTER")

    if after is None:
        return False

    final_error = after.center_x_px - TARGET_CENTER_X_PX

    print()
    print(f"Lateral before : {before.center_x_px:.2f} px")
    print(f"Command y      : {lateral_y_m:+.4f} m")
    print(f"Lateral after  : {after.center_x_px:.2f} px")
    print(f"Final error    : {final_error:+.2f} px")

    return abs(final_error) <= CENTER_X_TOL_PX


# ============================================================
# Camera / Main
# ============================================================


def start_camera(serial: str | None) -> rs.pipeline:
    """D435 RGB 카메라를 시작한다."""
    pipeline = rs.pipeline()
    config = rs.config()

    if serial:
        config.enable_device(serial)

    config.enable_stream(rs.stream.color, CAM_WIDTH, CAM_HEIGHT, rs.format.bgr8, CAM_FPS)
    pipeline.start(config)

    for _ in range(CAMERA_WARMUP_FRAMES):
        pipeline.wait_for_frames()

    print(f"D435 시작: {CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS}")

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=None, help="사용할 D435 serial")
    args = parser.parse_args()

    pipeline = start_camera(args.serial)

    robot = initialize_mobile(address=ADDRESS, model="m")
    monitor = OdometryMonitor()

    try:
        robot.start_state_update(monitor.on_state, rate=50)

        if not wait_for_odometry(monitor):
            print("Odometry를 받지 못했습니다.")
            return

        print()
        print("==========================================")
        print("       TOTE YAW + LATERAL ALIGN")
        print("==========================================")
        print(f"TARGET ANGLE = {TARGET_ANGLE_DEG:+.2f} deg")
        print(f"TARGET CX    = {TARGET_CENTER_X_PX:.2f} px")
        print(f"TARGET CY    = {TARGET_CENTER_Y_PX:.2f} px")
        print()

        if not align_yaw_once(robot, monitor, pipeline):
            print()
            print("RESULT: YAW FAILED")
            return

        time.sleep(SETTLE_S)
        flush_camera(pipeline)

        if not align_lateral_once(robot, monitor, pipeline):
            print()
            print("RESULT: LATERAL FAILED")
            return

        print()
        print("==========================================")
        print("RESULT: SUCCESS")
        print("==========================================")

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