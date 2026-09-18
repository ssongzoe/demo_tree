#!/usr/bin/env python3
"""D435 RGB 영상에서 새 head bracket용 tote TOP / corner를 검출한다.

역할
- RGB -> gray -> Gaussian blur -> Canny -> morphology close
- HoughLinesP로 TOP / LEFT / RIGHT line 후보 검출
- TOP과 side line 교점으로 TL / TR 계산
- 여러 frame의 LEFT feature를 median으로 측정

새 bracket 정지 테스트(640x480)에서 TOP은 y≈152 px, TL은 x≈70 px,
TR은 x≈588 px에 보였다. TOP / LEFT 후보가 각각 6개, RIGHT는 1개였으므로
RIGHT 교점이 좋은 TOP을 우선하고 corner의 frame 간 튀어오름을 제한한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np
import pyrealsense2 as rs


# 카메라
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30
CAMERA_WARMUP_FRAMES = 30
CAMERA_FLUSH_FRAMES = 10

# 공통 영상 처리
CANNY_LOW = 40
CANNY_HIGH = 120
HOUGH_THRESHOLD = 45
MAX_LINE_GAP = 40

# TOP rim
# 물건이 rim 가까이 올라오면 TOP edge가 여러 Hough segment로 끊길 수 있으므로
# 짧은 조각도 묶어 다시 fitting한다.
TOP_FRAGMENT_MIN_LINE_LENGTH = 60
TOP_MIN_FITTED_LENGTH = 100
# 새 bracket 기준 자세 TOP y≈152 px. +/-7 cm 이동 범위도 포함하되
# 천장/로봇 구조물이 많은 영상 상단은 제외한다.
TOP_MIN_Y = 80
TOP_MAX_Y = 200
TOP_MAX_ANGLE_DEG = 12.0

# Tote rim의 앞/뒷 edge를 하나의 group으로 섞지 않는다.
# 기존 10 px / 3 deg는 calibration 중 서로 다른 두 TOP을
# 한 fitLine에 섞는 원인이 될 수 있었다.
TOP_GROUP_Y_TOL_PX = 5.0
TOP_GROUP_ANGLE_TOL_DEG = 1.0

# 거의 같은 위치/각도에서 중복 생성된 fitted TOP 제거 기준
TOP_DUPLICATE_Y_TOL_PX = 2.0
TOP_DUPLICATE_ANGLE_TOL_DEG = 0.35

# 점수가 거의 같은 TOP 후보는 항상 위쪽 rim을 선택한다.
# 이렇게 해야 calibration과 실행 시의 선택이 재현된다.
TOP_STRONG_SCORE_MARGIN = 0.08

# 정지 구간에서 이미 선택한 TOP을 다른 rim으로 바꾸지 않기 위한 tracking.
# 로봇 이동 후 진짜 TOP이 크게 달라지면 아래 miss 횟수 후 재획득한다.
TOP_TRACK_Y_TOL_PX = 4.0
TOP_TRACK_ANGLE_TOL_DEG = 0.75
TOP_TRACK_REACQUIRE_MISSES = 6

# TOP 후보 점수에 corner geometry를 반영한다.
# 실험에서 RIGHT side 후보는 1개로 안정적이었으므로 가중치를 더 크게 둔다.
TOP_RIGHT_CORNER_BONUS = 2.0
TOP_LEFT_CORNER_BONUS = 0.7
TOP_BOTH_CORNERS_BONUS = 0.6

CONTRAST_OFFSET_PX = 10
CONTRAST_SAMPLE_COUNT = 20

# LEFT / RIGHT side
SIDE_MIN_LINE_LENGTH = 60

LEFT_MIN_ANGLE_DEG = 90.0
LEFT_MAX_ANGLE_DEG = 145.0

RIGHT_MIN_ANGLE_DEG = 35.0
RIGHT_MAX_ANGLE_DEG = 90.0

MAX_LEFT_CANDIDATES = 20
MAX_RIGHT_CANDIDATES = 20

# Corner geometry / edge support
CORNER_MARGIN_RATIO = 0.06
SIDE_PROBE_HEIGHT_RATIO = 0.30

# 내부 물건 edge가 side 후보로 들어와도 실제 tote 바깥 corner와 너무 멀면 제거한다.
LEFT_CORNER_MAX_X_RATIO = 0.40
RIGHT_CORNER_MIN_X_RATIO = 0.60
MIN_TOP_WIDTH_RATIO = 0.65
MAX_TOP_WIDTH_RATIO = 1.10

EDGE_SAMPLE_COUNT = 20
EDGE_SEARCH_RADIUS = 4
MIN_SIDE_EDGE_SUPPORT = 0.35

# corner 선택 / tracking
CORNER_TRACK_MAX_JUMP_PX = 18.0
CORNER_TRACK_REACQUIRE_MISSES = 6
CORNER_ENDPOINT_DISTANCE_PX = 60.0
CORNER_BOUNDARY_RATIO = 0.04

WINDOW_NAME = "Tote Vision"


@dataclass
class LineCandidate:
    """Hough 선분과 무한 직선 표현."""

    segment: np.ndarray
    length: float
    angle_deg: float
    center_x: float
    center_y: float
    line_abc: np.ndarray


@dataclass
class CornerFeature:
    """한쪽 TOP corner 측정값."""

    point: np.ndarray
    side_probe: np.ndarray
    side: LineCandidate


@dataclass
class FrameFeature:
    """한 frame에서 검출된 TOP / TL / TR."""

    top: LineCandidate
    top_angle_deg: float
    left: CornerFeature | None
    right: CornerFeature | None


@dataclass
class LeftMeasurement:
    """여러 frame에서 얻은 LEFT feature median."""

    tl_x_px: float
    tl_y_px: float
    angle_deg: float
    tl_x_std_px: float
    tl_y_std_px: float
    angle_std_deg: float
    valid_frames: int


@dataclass
class _TopHypothesis:
    """TOP 후보와 해당 frame의 기본 점수."""

    line: LineCandidate
    score: float


@dataclass
class _CornerHypothesis:
    """corner 후보와 side edge / endpoint geometry 점수."""

    corner: CornerFeature
    score: float


# detect_frame_feature() API는 그대로 두고 module 내부에서만 TOP을 추적한다.
_tracked_top: LineCandidate | None = None
_top_track_misses = 0
_tracked_left_point: np.ndarray | None = None
_tracked_right_point: np.ndarray | None = None
_left_track_misses = 0
_right_track_misses = 0
_diagnostic_frame_count = 0


def reset_top_tracking() -> None:
    """TOP / corner tracking 상태를 지운다."""
    global _tracked_top, _top_track_misses
    global _tracked_left_point, _tracked_right_point
    global _left_track_misses, _right_track_misses
    global _diagnostic_frame_count

    _tracked_top = None
    _top_track_misses = 0
    _tracked_left_point = None
    _tracked_right_point = None
    _left_track_misses = 0
    _right_track_misses = 0
    _diagnostic_frame_count = 0


def normalize_angle(angle_deg: float) -> float:
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

    return math.hypot(dx, dy), math.degrees(math.atan2(dy, dx)) % 180.0, 0.5 * (x1 + x2), 0.5 * (y1 + y2)


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


def intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
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
    return float(x) if np.isfinite(x) else None


def line_y_at_x(segment: np.ndarray, x: float) -> float | None:
    """선분의 무한 직선에서 지정 x 위치의 y를 계산한다."""
    x1, y1, x2, y2 = [float(value) for value in segment]

    if abs(x2 - x1) < 1e-8:
        return None

    return y1 + (float(x) - x1) * (y2 - y1) / (x2 - x1)


def angle_difference_deg(first: float, second: float) -> float:
    """수평 기준 두 line angle의 최소 차이를 계산한다."""
    return abs(normalize_angle(float(first) - float(second)))


def fit_top_group(group: list[LineCandidate]) -> LineCandidate | None:
    """같은 TOP rim의 Hough 조각 endpoint를 fitLine으로 합쳐 하나의 긴 TOP line을 만든다."""
    points = []

    for candidate in group:
        x1, y1, x2, y2 = candidate.segment
        points.append([x1, y1])
        points.append([x2, y2])

    if len(points) < 4:
        return None

    points = np.asarray(points, dtype=np.float32)
    vx, vy, fit_x, fit_y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)

    if abs(vx) < 1e-8:
        return None

    left_x = float(np.min(points[:, 0]))
    right_x = float(np.max(points[:, 0]))
    left_y = float(fit_y + (left_x - fit_x) * vy / vx)
    right_y = float(fit_y + (right_x - fit_x) * vy / vx)
    fitted = make_candidate(np.asarray([left_x, left_y, right_x, right_y], dtype=np.float64))

    if fitted.length < TOP_MIN_FITTED_LENGTH:
        return None

    return fitted


def preprocess(image: np.ndarray):
    """gray / edge 영상을 만든다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    return gray, edges


def top_contrast(gray: np.ndarray, segment: np.ndarray) -> float:
    """TOP rim 위/아래의 밝기 차이를 측정한다."""
    height, width = gray.shape
    x1, _, x2, _ = segment
    xs = np.linspace(min(x1, x2), max(x1, x2), CONTRAST_SAMPLE_COUNT)

    above = []
    below = []

    for x_value in xs:
        y_value = line_y_at_x(segment, x_value)

        if y_value is None:
            continue

        x = int(round(x_value))
        y = int(round(y_value))
        y_above = y - CONTRAST_OFFSET_PX
        y_below = y + CONTRAST_OFFSET_PX

        if not (1 <= x < width - 1 and 1 <= y_above < height - 1 and 1 <= y_below < height - 1):
            continue

        above.append(float(np.mean(gray[y_above - 1:y_above + 2, x - 1:x + 2])))
        below.append(float(np.mean(gray[y_below - 1:y_below + 2, x - 1:x + 2])))

    if not above:
        return -1000.0

    return float(np.mean(above) - np.mean(below))


def _same_top_group(first: LineCandidate, second: LineCandidate) -> bool:
    """Hough 조각 두 개가 같은 TOP edge인지 판정한다."""
    if angle_difference_deg(first.angle_deg, second.angle_deg) > TOP_GROUP_ANGLE_TOL_DEG:
        return False

    # 영상 중앙만 비교하면 중앙에서 교차하는 서로 다른 rim이
    # 한 group으로 섞일 수 있다. 세 지점을 모두 비교한다.
    for x_value in (CAM_WIDTH * 0.20, CAM_WIDTH * 0.50, CAM_WIDTH * 0.80):
        first_y = line_y_at_x(first.segment, x_value)
        second_y = line_y_at_x(second.segment, x_value)

        if first_y is None or second_y is None:
            return False

        if abs(first_y - second_y) > TOP_GROUP_Y_TOL_PX:
            return False

    return True


def _top_descriptor(line: LineCandidate) -> tuple[float, float]:
    """TOP 비교용 (영상 중앙 y, 정규화 angle)을 반환한다."""
    center_y = line_y_at_x(line.segment, CAM_WIDTH * 0.50)

    if center_y is None:
        center_y = line.center_y

    return float(center_y), normalize_angle(line.angle_deg)


def _is_duplicate_top(first: LineCandidate, second: LineCandidate) -> bool:
    """anchor만 다르고 실제로는 같은 fitted TOP인지 판정한다."""
    first_y, first_angle = _top_descriptor(first)
    second_y, second_angle = _top_descriptor(second)

    return (
        abs(first_y - second_y) <= TOP_DUPLICATE_Y_TOL_PX
        and angle_difference_deg(first_angle, second_angle) <= TOP_DUPLICATE_ANGLE_TOL_DEG
    )


def _detect_top_hypotheses(gray: np.ndarray, edges: np.ndarray) -> list[_TopHypothesis]:
    """TOP 조각을 섞지 않고 별도의 fitted line 후보로 반환한다."""
    roi_edges = np.zeros_like(edges)
    roi_edges[TOP_MIN_Y:TOP_MAX_Y, :] = edges[TOP_MIN_Y:TOP_MAX_Y, :]

    detected = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=HOUGH_THRESHOLD,
        minLineLength=TOP_FRAGMENT_MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    if detected is None:
        return []

    candidates = []

    for segment in np.asarray(detected, dtype=np.int32).reshape(-1, 4):
        candidate = make_candidate(segment)
        angle_deg = normalize_angle(candidate.angle_deg)

        if abs(angle_deg) > TOP_MAX_ANGLE_DEG or not TOP_MIN_Y <= candidate.center_y <= TOP_MAX_Y:
            continue

        candidates.append(candidate)

    if not candidates:
        return []

    hypotheses: list[_TopHypothesis] = []

    for anchor in candidates:
        group = [candidate for candidate in candidates if _same_top_group(anchor, candidate)]

        fitted = fit_top_group(group)

        if fitted is None:
            continue

        angle_deg = normalize_angle(fitted.angle_deg)

        if abs(angle_deg) > TOP_MAX_ANGLE_DEG or not TOP_MIN_Y <= fitted.center_y <= TOP_MAX_Y:
            continue

        coverage_score = min(1.0, fitted.length / (CAM_WIDTH * 0.80))
        fragment_score = min(1.0, sum(item.length for item in group) / CAM_WIDTH)
        contrast_score = np.clip(top_contrast(gray, fitted.segment) / 80.0, -1.0, 1.0)
        group_score = min(1.0, len(group) / 4.0)
        score = 0.50 * coverage_score + 0.20 * fragment_score + 0.20 * contrast_score + 0.10 * group_score

        duplicate_index = next(
            (index for index, item in enumerate(hypotheses) if _is_duplicate_top(fitted, item.line)),
            None,
        )

        hypothesis = _TopHypothesis(line=fitted, score=float(score))

        if duplicate_index is None:
            hypotheses.append(hypothesis)
        elif score > hypotheses[duplicate_index].score:
            hypotheses[duplicate_index] = hypothesis

    return hypotheses


def _select_top(hypotheses: list[_TopHypothesis]) -> LineCandidate | None:
    """deterministic 선택과 short-term tracking으로 TOP rim mode switching을 막는다."""
    global _tracked_top, _top_track_misses

    if not hypotheses:
        _top_track_misses += 1

        if _top_track_misses >= TOP_TRACK_REACQUIRE_MISSES:
            _tracked_top = None

        return None

    if _tracked_top is not None:
        tracked_y, tracked_angle = _top_descriptor(_tracked_top)
        matching: list[tuple[float, float, _TopHypothesis]] = []

        for hypothesis in hypotheses:
            candidate_y, candidate_angle = _top_descriptor(hypothesis.line)
            y_delta = abs(candidate_y - tracked_y)
            angle_delta = angle_difference_deg(candidate_angle, tracked_angle)

            if y_delta <= TOP_TRACK_Y_TOL_PX and angle_delta <= TOP_TRACK_ANGLE_TOL_DEG:
                # 연속성을 점수보다 우선해 인접한 다른 rim으로의 전환을 막는다.
                continuity_cost = y_delta / TOP_TRACK_Y_TOL_PX + angle_delta / TOP_TRACK_ANGLE_TOL_DEG
                matching.append((continuity_cost, -hypothesis.score, hypothesis))

        if matching:
            selected = min(matching, key=lambda item: (item[0], item[1]))[2].line
            _tracked_top = selected
            _top_track_misses = 0
            return selected

        # 로봇 이동 직후에는 기존 선을 즉시 다른 rim으로 바꾸지 않는다.
        _top_track_misses += 1

        if _top_track_misses < TOP_TRACK_REACQUIRE_MISSES:
            return None

        _tracked_top = None

    best_score = max(item.score for item in hypotheses)
    strong = [item for item in hypotheses if item.score >= best_score - TOP_STRONG_SCORE_MARGIN]

    # 허용 점수 내에서는 항상 영상 상단의 rim을 선택해
    # 실행할 때마다 앞/뒷 edge가 바뀌는 것을 막는다.
    selected_hypothesis = min(
        strong,
        key=lambda item: (_top_descriptor(item.line)[0], -item.score),
    )
    _tracked_top = selected_hypothesis.line
    _top_track_misses = 0

    return selected_hypothesis.line


def detect_top(gray: np.ndarray, edges: np.ndarray) -> LineCandidate | None:
    """TOP 후보를 분리하고 frame 간 일관된 하나의 rim을 선택한다."""
    hypotheses = _detect_top_hypotheses(gray, edges)

    return _select_top(hypotheses)


def detect_side_candidates(edges: np.ndarray):
    """LEFT / RIGHT side 후보를 분리한다."""
    _, width = edges.shape

    detected = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=HOUGH_THRESHOLD,
        minLineLength=SIDE_MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    if detected is None:
        return [], []

    left_candidates = []
    right_candidates = []

    for segment in np.asarray(detected, dtype=np.int32).reshape(-1, 4):
        candidate = make_candidate(segment)

        if LEFT_MIN_ANGLE_DEG <= candidate.angle_deg <= LEFT_MAX_ANGLE_DEG and candidate.center_x < width * 0.70:
            left_candidates.append(candidate)
        elif RIGHT_MIN_ANGLE_DEG <= candidate.angle_deg <= RIGHT_MAX_ANGLE_DEG and candidate.center_x > width * 0.30:
            right_candidates.append(candidate)

    left_candidates.sort(key=lambda item: item.length, reverse=True)
    right_candidates.sort(key=lambda item: item.length, reverse=True)

    return left_candidates[:MAX_LEFT_CANDIDATES], right_candidates[:MAX_RIGHT_CANDIDATES]


def edge_support(edges: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """예상 side 주변에 실제 edge가 얼마나 존재하는지 계산한다."""
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


def valid_corner(point: np.ndarray, width: int, height: int) -> bool:
    """corner 교점이 영상 내부 또는 약간 바깥에 있는지 확인한다."""
    margin_x = width * CORNER_MARGIN_RATIO
    margin_y = height * CORNER_MARGIN_RATIO

    return -margin_x <= point[0] <= width + margin_x and -margin_y <= point[1] <= height + margin_y


def build_corner(top: LineCandidate, side: LineCandidate, edges: np.ndarray, is_left: bool) -> CornerFeature | None:
    """TOP과 side의 교점으로 한쪽 corner를 만든다."""
    height, width = edges.shape
    point = intersection(top.line_abc, side.line_abc)

    if point is None or not valid_corner(point, width, height):
        return None

    if is_left and point[0] > width * LEFT_CORNER_MAX_X_RATIO:
        return None

    if not is_left and point[0] < width * RIGHT_CORNER_MIN_X_RATIO:
        return None

    probe_y = min(height * 0.90, point[1] + height * SIDE_PROBE_HEIGHT_RATIO)
    probe_x = line_x_at_y(side.line_abc, probe_y)

    if probe_x is None:
        return None

    side_probe = np.asarray([probe_x, probe_y], dtype=np.float64)

    if is_left and side_probe[0] >= point[0]:
        return None

    if not is_left and side_probe[0] <= point[0]:
        return None

    if edge_support(edges, point, side_probe) < MIN_SIDE_EDGE_SUPPORT:
        return None

    return CornerFeature(point=point, side_probe=side_probe, side=side)


def _corner_quality(corner: CornerFeature, edges: np.ndarray, is_left: bool) -> float:
    """side edge가 실제 corner에서 시작하는지까지 포함해 점수화한다."""
    height, width = edges.shape
    support = edge_support(edges, corner.point, corner.side_probe)
    endpoints = corner.side.segment.reshape(2, 2)
    endpoint_distance = min(
        float(np.linalg.norm(corner.point - endpoint))
        for endpoint in endpoints
    )
    endpoint_score = max(0.0, 1.0 - endpoint_distance / CORNER_ENDPOINT_DISTANCE_PX)
    length_score = min(1.0, corner.side.length / (height * 0.35))

    x_ratio = float(corner.point[0]) / width
    boundary_penalty = 0.0
    if is_left and x_ratio < CORNER_BOUNDARY_RATIO:
        boundary_penalty = (CORNER_BOUNDARY_RATIO - x_ratio) / CORNER_BOUNDARY_RATIO
    elif not is_left and x_ratio > 1.0 - CORNER_BOUNDARY_RATIO:
        boundary_penalty = (x_ratio - (1.0 - CORNER_BOUNDARY_RATIO)) / CORNER_BOUNDARY_RATIO

    return 1.5 * support + 0.9 * endpoint_score + 0.4 * length_score - 1.5 * boundary_penalty


def _corner_hypotheses(
    top: LineCandidate,
    candidates: list[LineCandidate],
    edges: np.ndarray,
    *,
    is_left: bool,
) -> list[_CornerHypothesis]:
    hypotheses: list[_CornerHypothesis] = []

    for side in candidates:
        corner = build_corner(top, side, edges, is_left=is_left)

        if corner is None:
            continue

        hypotheses.append(
            _CornerHypothesis(
                corner=corner,
                score=_corner_quality(corner, edges, is_left),
            )
        )

    return hypotheses


def _best_corner(hypotheses: list[_CornerHypothesis]) -> _CornerHypothesis | None:
    return max(hypotheses, key=lambda item: item.score) if hypotheses else None


def _track_corner(
    hypotheses: list[_CornerHypothesis],
    *,
    is_left: bool,
) -> CornerFeature | None:
    """static 구간에서 corner가 인접한 다른 side로 튀는 것을 막는다."""
    global _tracked_left_point, _tracked_right_point
    global _left_track_misses, _right_track_misses

    tracked_point = _tracked_left_point if is_left else _tracked_right_point
    misses = _left_track_misses if is_left else _right_track_misses

    if tracked_point is not None and hypotheses:
        matching: list[tuple[float, float, _CornerHypothesis]] = []

        for hypothesis in hypotheses:
            distance = float(np.linalg.norm(hypothesis.corner.point - tracked_point))

            if distance <= CORNER_TRACK_MAX_JUMP_PX:
                matching.append((distance, -hypothesis.score, hypothesis))

        if matching:
            selected = min(matching, key=lambda item: (item[0], item[1]))[2].corner

            if is_left:
                _tracked_left_point = selected.point.copy()
                _left_track_misses = 0
            else:
                _tracked_right_point = selected.point.copy()
                _right_track_misses = 0

            return selected

    if tracked_point is not None:
        misses += 1

        if is_left:
            _left_track_misses = misses
        else:
            _right_track_misses = misses

        if misses < CORNER_TRACK_REACQUIRE_MISSES:
            return None

        tracked_point = None
        if is_left:
            _tracked_left_point = None
        else:
            _tracked_right_point = None

    if not hypotheses:
        return None

    selected = _best_corner(hypotheses).corner
    if is_left:
        _tracked_left_point = selected.point.copy()
        _left_track_misses = 0
    else:
        _tracked_right_point = selected.point.copy()
        _right_track_misses = 0

    return selected


def choose_left_corner(top: LineCandidate, candidates: list[LineCandidate], edges: np.ndarray) -> CornerFeature | None:
    """LEFT 후보 중 edge / endpoint geometry 점수가 가장 좋은 TL을 선택한다."""
    best = _best_corner(_corner_hypotheses(top, candidates, edges, is_left=True))
    return best.corner if best is not None else None


def choose_right_corner(top: LineCandidate, candidates: list[LineCandidate], edges: np.ndarray) -> CornerFeature | None:
    """RIGHT 후보 중 edge / endpoint geometry 점수가 가장 좋은 TR을 선택한다."""
    best = _best_corner(_corner_hypotheses(top, candidates, edges, is_left=False))
    return best.corner if best is not None else None


def _score_top_with_corners(
    hypotheses: list[_TopHypothesis],
    left_candidates: list[LineCandidate],
    right_candidates: list[LineCandidate],
    edges: np.ndarray,
) -> list[_TopHypothesis]:
    """TOP이 실제 tote side와 만드는 corner geometry를 점수에 합친다."""
    scored: list[_TopHypothesis] = []

    for hypothesis in hypotheses:
        left = _best_corner(
            _corner_hypotheses(hypothesis.line, left_candidates, edges, is_left=True)
        )
        right = _best_corner(
            _corner_hypotheses(hypothesis.line, right_candidates, edges, is_left=False)
        )
        score = hypothesis.score

        if right is not None:
            score += TOP_RIGHT_CORNER_BONUS + 0.5 * right.score
        if left is not None:
            score += TOP_LEFT_CORNER_BONUS + 0.2 * left.score

        if left is not None and right is not None:
            width = float(np.linalg.norm(right.corner.point - left.corner.point))
            if CAM_WIDTH * MIN_TOP_WIDTH_RATIO <= width <= CAM_WIDTH * MAX_TOP_WIDTH_RATIO:
                score += TOP_BOTH_CORNERS_BONUS
            else:
                score -= TOP_BOTH_CORNERS_BONUS

        scored.append(_TopHypothesis(line=hypothesis.line, score=score))

    return scored


def _right_rejection_summary(
    top: LineCandidate,
    candidates: list[LineCandidate],
    edges: np.ndarray,
) -> tuple[dict[str, int], str]:
    """RIGHT raw 후보가 build_corner()에서 탈락한 이유를 진단한다."""
    height, width = edges.shape
    reasons: dict[str, int] = {}
    descriptions = []

    for candidate in candidates:
        reason = "valid"
        point = intersection(top.line_abc, candidate.line_abc)
        support = None

        if point is None:
            reason = "parallel"
        elif not valid_corner(point, width, height):
            reason = "outside_image"
        elif point[0] < width * RIGHT_CORNER_MIN_X_RATIO:
            reason = "corner_too_left"
        else:
            probe_y = min(
                height * 0.90,
                point[1] + height * SIDE_PROBE_HEIGHT_RATIO,
            )
            probe_x = line_x_at_y(candidate.line_abc, probe_y)

            if probe_x is None:
                reason = "no_probe"
            elif probe_x <= point[0]:
                reason = "wrong_direction"
            else:
                side_probe = np.asarray([probe_x, probe_y], dtype=np.float64)
                support = edge_support(edges, point, side_probe)

                if support < MIN_SIDE_EDGE_SUPPORT:
                    reason = "low_edge_support"

        reasons[reason] = reasons.get(reason, 0) + 1
        point_text = "None" if point is None else f"({point[0]:.1f},{point[1]:.1f})"
        support_text = "--" if support is None else f"{support:.2f}"
        descriptions.append(
            f"len={candidate.length:.1f},ang={candidate.angle_deg:.1f},"
            f"p={point_text},support={support_text},reason={reason}"
        )

    return reasons, " | ".join(descriptions)


def _print_right_diagnostic(
    top: LineCandidate,
    right_candidates: list[LineCandidate],
    right_hypotheses: list[_CornerHypothesis],
    edges: np.ndarray,
) -> None:
    """TR 미검출이 Hough, geometry, tracking 중 어디서 발생했는지 출력한다."""
    tracked_text = (
        "None"
        if _tracked_right_point is None
        else f"({_tracked_right_point[0]:.1f},{_tracked_right_point[1]:.1f})"
    )

    if not right_candidates:
        print(
            f"[RIGHT_DIAG frame={_diagnostic_frame_count}] stage=HOUGH "
            f"RAW_RIGHT=0 VALID_RIGHT=0 tracked={tracked_text} "
            f"misses={_right_track_misses}"
        )
        return

    if not right_hypotheses:
        reasons, descriptions = _right_rejection_summary(
            top,
            right_candidates,
            edges,
        )
        print(
            f"[RIGHT_DIAG frame={_diagnostic_frame_count}] stage=GEOMETRY "
            f"RAW_RIGHT={len(right_candidates)} VALID_RIGHT=0 "
            f"tracked={tracked_text} misses={_right_track_misses} "
            f"reject={reasons} candidates=[{descriptions}]"
        )
        return

    points = [hypothesis.corner.point for hypothesis in right_hypotheses]
    point_text = ",".join(f"({point[0]:.1f},{point[1]:.1f})" for point in points)
    nearest_jump = (
        None
        if _tracked_right_point is None
        else min(float(np.linalg.norm(point - _tracked_right_point)) for point in points)
    )
    jump_text = "--" if nearest_jump is None else f"{nearest_jump:.1f}px"
    print(
        f"[RIGHT_DIAG frame={_diagnostic_frame_count}] stage=TRACKING "
        f"RAW_RIGHT={len(right_candidates)} VALID_RIGHT={len(right_hypotheses)} "
        f"tracked={tracked_text} misses={_right_track_misses} "
        f"nearest_jump={jump_text} valid_points=[{point_text}]"
    )


def detect_frame_feature_with_edges(
    image: np.ndarray,
) -> tuple[FrameFeature | None, np.ndarray]:
    """TOP / corner와 같은 frame에서 사용한 Canny edge 영상을 반환한다."""
    global _diagnostic_frame_count

    _diagnostic_frame_count += 1
    gray, edges = preprocess(image)
    left_candidates, right_candidates = detect_side_candidates(edges)
    top_hypotheses = _detect_top_hypotheses(gray, edges)
    scored_top_hypotheses = _score_top_with_corners(
        top_hypotheses,
        left_candidates,
        right_candidates,
        edges,
    )
    top = _select_top(scored_top_hypotheses)

    if top is None:
        _track_corner([], is_left=True)
        _track_corner([], is_left=False)
        print(
            f"[RIGHT_DIAG frame={_diagnostic_frame_count}] stage=NO_TOP "
            f"RAW_RIGHT={len(right_candidates)} VALID_RIGHT=0"
        )
        return None, edges

    left_hypotheses = _corner_hypotheses(top, left_candidates, edges, is_left=True)
    right_hypotheses = _corner_hypotheses(top, right_candidates, edges, is_left=False)
    left = _track_corner(left_hypotheses, is_left=True)
    right = _track_corner(right_hypotheses, is_left=False)

    if right is None:
        _print_right_diagnostic(top, right_candidates, right_hypotheses, edges)

    # width가 비정상이면 실험에서 안정적이었던 RIGHT는 유지하고
    # 잘못된 LEFT 후보만 제거한다.
    if left is not None and right is not None:
        top_width = float(np.linalg.norm(right.point - left.point))

        if not CAM_WIDTH * MIN_TOP_WIDTH_RATIO <= top_width <= CAM_WIDTH * MAX_TOP_WIDTH_RATIO:
            left = None

    feature = FrameFeature(
        top=top,
        top_angle_deg=normalize_angle(top.angle_deg),
        left=left,
        right=right,
    )

    return feature, edges


def detect_frame_feature(image: np.ndarray) -> FrameFeature | None:
    """기존 alignment 코드와 호환되도록 TOP / TL / TR feature만 반환한다."""
    feature, _ = detect_frame_feature_with_edges(image)
    return feature


def draw_feature(image: np.ndarray, feature: FrameFeature | None, label: str = "") -> np.ndarray:
    """TOP / TL / TR 검출 결과를 표시한다."""
    output = image.copy()

    if feature is None:
        cv2.putText(output, "TOP NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return output

    p1 = tuple(np.rint(feature.top.segment[:2]).astype(int))
    p2 = tuple(np.rint(feature.top.segment[2:]).astype(int))
    cv2.line(output, p1, p2, (0, 255, 255), 4, cv2.LINE_AA)

    if feature.left is not None:
        tl = tuple(np.rint(feature.left.point).astype(int))
        probe = tuple(np.rint(feature.left.side_probe).astype(int))
        cv2.line(output, tl, probe, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(output, tl, 8, (0, 0, 255), -1)
        cv2.putText(output, "TL", (tl[0] + 8, tl[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    if feature.right is not None:
        tr = tuple(np.rint(feature.right.point).astype(int))
        probe = tuple(np.rint(feature.right.side_probe).astype(int))
        cv2.line(output, tr, probe, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(output, tr, 8, (255, 0, 255), -1)
        cv2.putText(output, "TR", (tr[0] - 35, tr[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)

    left_text = "TL=OK" if feature.left is not None else "TL=--"
    right_text = "TR=OK" if feature.right is not None else "TR=--"
    text = f"angle={feature.top_angle_deg:+.2f}  {left_text}  {right_text}"
    cv2.putText(output, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)

    if label:
        cv2.putText(output, label, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)

    return output


def start_camera(serial: str | None) -> rs.pipeline:
    """D435 RGB 카메라를 시작한다."""
    reset_top_tracking()

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


def flush_camera(pipeline: rs.pipeline, frame_count: int = CAMERA_FLUSH_FRAMES) -> None:
    """로봇 이동 중 쌓인 이전 frame을 버린다."""
    for _ in range(frame_count):
        pipeline.wait_for_frames()


def measure_left_feature(
    pipeline: rs.pipeline,
    *,
    frame_count: int = 40,
    timeout_s: float = 10.0,
    show: bool = False,
    label: str = "",
) -> LeftMeasurement | None:
    """TOP + TL이 동시에 검출된 frame을 모아 median feature를 반환한다."""
    tl_xs: list[float] = []
    tl_ys: list[float] = []
    angles: list[float] = []

    start_time = time.monotonic()

    while len(tl_xs) < frame_count and time.monotonic() - start_time < timeout_s:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asarray(color_frame.get_data())
        feature = detect_frame_feature(image)

        if show:
            cv2.imshow(WINDOW_NAME, draw_feature(image, feature, label=label))
            cv2.waitKey(1)

        if feature is None or feature.left is None:
            continue

        tl_xs.append(float(feature.left.point[0]))
        tl_ys.append(float(feature.left.point[1]))
        angles.append(float(feature.top_angle_deg))

    if len(tl_xs) < frame_count // 2:
        return None

    return LeftMeasurement(
        tl_x_px=float(np.median(tl_xs)),
        tl_y_px=float(np.median(tl_ys)),
        angle_deg=float(np.median(angles)),
        tl_x_std_px=float(np.std(tl_xs)),
        tl_y_std_px=float(np.std(tl_ys)),
        angle_std_deg=float(np.std(angles)),
        valid_frames=len(tl_xs),
    )