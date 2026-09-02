#!/usr/bin/env python3
"""D435 RGB 영상에서 tote의 TOP rim과 좌/우 상단 corner를 검출한다.

역할
- RGB -> gray -> Gaussian blur -> Canny -> morphology close
- HoughLinesP로 TOP / LEFT / RIGHT line 후보 검출
- TOP과 side line 교점으로 TL / TR 계산
- 여러 frame의 LEFT feature를 median으로 측정

※ 물건이 찬 tote에서도 안정적으로 동작하도록
   Robust calibration test의 검출 방식을 사용한다.
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
TOP_MIN_Y = 55
TOP_MAX_Y = 175
TOP_MAX_ANGLE_DEG = 12.0

TOP_GROUP_Y_TOL_PX = 10.0
TOP_GROUP_ANGLE_TOL_DEG = 3.0

CONTRAST_OFFSET_PX = 10
CONTRAST_SAMPLE_COUNT = 20

# LEFT / RIGHT side
SIDE_MIN_LINE_LENGTH = 60

LEFT_MIN_ANGLE_DEG = 90.0
LEFT_MAX_ANGLE_DEG = 145.0

RIGHT_MIN_ANGLE_DEG = 35.0
RIGHT_MAX_ANGLE_DEG = 90.0

MAX_LEFT_CANDIDATES = 6
MAX_RIGHT_CANDIDATES = 6

# Corner geometry / edge support
CORNER_MARGIN_RATIO = 0.06
SIDE_PROBE_HEIGHT_RATIO = 0.30

# 내부 물건 edge가 side 후보로 들어와도 실제 tote 바깥 corner와 너무 멀면 제거한다.
LEFT_CORNER_MAX_X_RATIO = 0.40
RIGHT_CORNER_MIN_X_RATIO = 0.60
MIN_TOP_WIDTH_RATIO = 0.55
MAX_TOP_WIDTH_RATIO = 1.10

EDGE_SAMPLE_COUNT = 20
EDGE_SEARCH_RADIUS = 4
MIN_SIDE_EDGE_SUPPORT = 0.35

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


def detect_top(gray: np.ndarray, edges: np.ndarray) -> LineCandidate | None:
    """TOP 조각을 y/angle 기준으로 묶고 fitLine으로 합쳐 가장 안정적인 TOP rim을 선택한다."""
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
        return None

    candidates = []

    for segment in np.asarray(detected, dtype=np.int32).reshape(-1, 4):
        candidate = make_candidate(segment)
        angle_deg = normalize_angle(candidate.angle_deg)

        if abs(angle_deg) > TOP_MAX_ANGLE_DEG or not TOP_MIN_Y <= candidate.center_y <= TOP_MAX_Y:
            continue

        candidates.append(candidate)

    if not candidates:
        return None

    best = None
    best_score = -float("inf")
    image_center_x = 0.5 * CAM_WIDTH

    for anchor in candidates:
        anchor_y = line_y_at_x(anchor.segment, image_center_x)

        if anchor_y is None:
            continue

        group = []

        for candidate in candidates:
            candidate_y = line_y_at_x(candidate.segment, image_center_x)

            if candidate_y is None:
                continue

            if abs(candidate_y - anchor_y) > TOP_GROUP_Y_TOL_PX:
                continue

            if angle_difference_deg(candidate.angle_deg, anchor.angle_deg) > TOP_GROUP_ANGLE_TOL_DEG:
                continue

            group.append(candidate)

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

        if score > best_score:
            best = fitted
            best_score = score

    return best


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


def choose_left_corner(top: LineCandidate, candidates: list[LineCandidate], edges: np.ndarray) -> CornerFeature | None:
    """유효한 LEFT 후보 중 가장 바깥쪽 TL을 선택한다."""
    valid = []

    for side in candidates:
        corner = build_corner(top, side, edges, is_left=True)

        if corner is not None:
            valid.append(corner)

    return min(valid, key=lambda item: item.point[0]) if valid else None


def choose_right_corner(top: LineCandidate, candidates: list[LineCandidate], edges: np.ndarray) -> CornerFeature | None:
    """유효한 RIGHT 후보 중 가장 바깥쪽 TR을 선택한다."""
    valid = []

    for side in candidates:
        corner = build_corner(top, side, edges, is_left=False)

        if corner is not None:
            valid.append(corner)

    return max(valid, key=lambda item: item.point[0]) if valid else None


def detect_frame_feature(image: np.ndarray) -> FrameFeature | None:
    """TOP을 먼저 찾고 LEFT / RIGHT corner를 서로 독립적으로 검출한다."""
    gray, edges = preprocess(image)
    top = detect_top(gray, edges)

    if top is None:
        return None

    left_candidates, right_candidates = detect_side_candidates(edges)
    left = choose_left_corner(top, left_candidates, edges)
    right = choose_right_corner(top, right_candidates, edges)

    # 양쪽 corner가 보일 때 실제 tote TOP 폭과 너무 다른 조합은
    # 내부 물건 edge 오검출로 보고 버린다.
    if left is not None and right is not None:
        top_width = float(np.linalg.norm(right.point - left.point))

        if not CAM_WIDTH * MIN_TOP_WIDTH_RATIO <= top_width <= CAM_WIDTH * MAX_TOP_WIDTH_RATIO:
            left = None
            right = None

    return FrameFeature(
        top=top,
        top_angle_deg=normalize_angle(top.angle_deg),
        left=left,
        right=right,
    )


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