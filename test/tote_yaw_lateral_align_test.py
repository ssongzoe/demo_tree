#!/usr/bin/env python3
"""토트 TOP rim으로 yaw를 맞춘 뒤 FINE center로 lateral / forward를 맞춘다.

정렬 기준
- Yaw: ±1도
- Lateral: ±2 cm
- Forward: ±2 cm

동작
1. TOP angle 측정 -> yaw 한 번 보정
2. TOP + LEFT + RIGHT -> center_x 측정 -> lateral 한 번 보정
3. FINE center_y 측정 -> forward 한 번 보정
4. 각 단계는 이동 후 한 번만 검증하며 반복 보정하지 않는다.

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.mobile_controller import OdometryMonitor, build_leg, initialize_mobile, move_leg, odom_pose, wait_for_odometry

# 로봇

ADDRESS = "192.168.30.1:50051"

SETTLE_S = 0.7

ALIGN_ANGULAR_SPEED = 0.5
ALIGN_LINEAR_SPEED = 0.08

QUINTIC_PEAK = 1.875
MIN_LEG_TIME = 1.5

# 정렬 기준 / Calibration

TARGET_ANGLE_DEG = 0.0
TARGET_CENTER_X_PX = 339.36
TARGET_CENTER_Y_PX = 105.50

ANGLE_TOL_DEG = 1.0
POSITION_TOL_M = 0.02

YAW_GAIN = 1.475
YAW_SIGN = -1.0

LATERAL_M_PER_PX = 0.00120
FORWARD_M_PER_PX = 0.00180

MAX_YAW_COMMAND_DEG = 10.0
MAX_LATERAL_COMMAND_M = 0.04
MAX_FORWARD_COMMAND_M = 0.04

# 카메라

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

CAMERA_WARMUP_FRAMES = 30
CAMERA_FLUSH_FRAMES = 10

# 영상 처리

CANNY_LOW = 40
CANNY_HIGH = 120

HOUGH_THRESHOLD = 45
MAX_LINE_GAP = 40

TOP_MIN_LINE_LENGTH = 100
SIDE_MIN_LINE_LENGTH = 60

TOP_MIN_Y = 55
TOP_MAX_Y = 175
TOP_MAX_ANGLE_DEG = 12.0

LEFT_MIN_ANGLE_DEG = 90.0
LEFT_MAX_ANGLE_DEG = 145.0

RIGHT_MIN_ANGLE_DEG = 35.0
RIGHT_MAX_ANGLE_DEG = 90.0

MAX_LEFT_CANDIDATES = 4
MAX_RIGHT_CANDIDATES = 4

CONTRAST_OFFSET_PX = 10
CONTRAST_SAMPLE_COUNT = 20

CORNER_MARGIN_RATIO = 0.06
MIN_TOP_WIDTH_RATIO = 0.45
MAX_TOP_WIDTH_RATIO = 1.10

SIDE_PROBE_HEIGHT_RATIO = 0.30

EDGE_SAMPLE_COUNT = 20
EDGE_SEARCH_RADIUS = 4

MIN_TOP_EDGE_SUPPORT = 0.55
MIN_SIDE_EDGE_SUPPORT = 0.35

# 여러 frame 측정

MEASURE_FRAMES = 20
MEASURE_TIMEOUT_S = 5.0

TOP_Y_FILTER_PX = 5.0
FINE_Y_FILTER_PX = 5.0
FINE_WIDTH_FILTER_PX = 10.0

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
class TopMeasurement:
    """여러 frame에서 측정한 TOP 결과."""

    angle_deg: float
    center_x_px: float
    center_y_px: float
    angle_std_deg: float
    center_y_std_px: float
    valid_frames: int

@dataclass
class FineFeature:
    """TOP과 양쪽 side의 교점으로 계산한 FINE feature."""

    angle_deg: float
    center_x_px: float
    center_y_px: float
    top_width_px: float
    tl: np.ndarray
    tr: np.ndarray
    left_probe: np.ndarray
    right_probe: np.ndarray

@dataclass
class FineMeasurement:
    """여러 frame에서 측정한 FINE 결과."""

    angle_deg: float
    center_x_px: float
    center_y_px: float
    top_width_px: float
    center_x_std_px: float
    center_y_std_px: float
    valid_frames: int

def clamp(value: float, minimum: float, maximum: float) -> float:
    """값을 지정 범위로 제한한다."""
    return max(minimum, min(maximum, value))

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
    x = -(b * y + c) / a
    if not np.isfinite(x):
        return None
    return float(x)

def line_y_at_x(segment: np.ndarray, x: float) -> float | None:
    """선분의 무한 직선에서 지정 x 위치의 y를 계산한다."""
    x1, y1, x2, y2 = [float(value) for value in segment]
    if abs(x2 - x1) < 1e-8:
        return None
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

def preprocess(image: np.ndarray):
    """모든 detector가 동일한 edge 영상을 사용한다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return gray, edges

# TOP detector

def top_contrast(gray: np.ndarray, segment: np.ndarray) -> float:
    """TOP 위쪽과 아래쪽 밝기 차이를 계산한다."""
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
    """Yaw와 FINE에서 공통으로 사용할 TOP rim 하나를 선택한다."""
    roi_edges = np.zeros_like(edges)
    roi_edges[TOP_MIN_Y:TOP_MAX_Y, :] = edges[TOP_MIN_Y:TOP_MAX_Y, :]
    detected = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=HOUGH_THRESHOLD,
        minLineLength=TOP_MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )
    if detected is None:
        return None
    best = None
    best_score = -float("inf")
    for segment in np.asarray(detected, dtype=np.int32).reshape(-1, 4):
        candidate = make_candidate(segment)
        angle_deg = normalize_angle(candidate.angle_deg)
        if abs(angle_deg) > TOP_MAX_ANGLE_DEG:
            continue
        if not TOP_MIN_Y <= candidate.center_y <= TOP_MAX_Y:
            continue
        contrast = top_contrast(gray, candidate.segment)
        length_score = candidate.length / CAM_WIDTH
        contrast_score = np.clip(contrast / 80.0, -1.0, 1.0)
        score = 0.70 * length_score + 0.30 * contrast_score
        if score > best_score:
            best = candidate
            best_score = score
    return best

# FINE detector

def detect_side_candidates(edges: np.ndarray):
    """LEFT / RIGHT rim 후보만 검출한다. TOP은 여기서 다시 찾지 않는다."""
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
        if LEFT_MIN_ANGLE_DEG <= candidate.angle_deg <= LEFT_MAX_ANGLE_DEG and candidate.center_x < width * 0.65:
            left_candidates.append(candidate)
        elif RIGHT_MIN_ANGLE_DEG <= candidate.angle_deg <= RIGHT_MAX_ANGLE_DEG and candidate.center_x > width * 0.35:
            right_candidates.append(candidate)
    left_candidates.sort(key=lambda item: item.length, reverse=True)
    right_candidates.sort(key=lambda item: item.length, reverse=True)
    return left_candidates[:MAX_LEFT_CANDIDATES], right_candidates[:MAX_RIGHT_CANDIDATES]

def edge_support(edges: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """예상 직선 주변에 실제 edge가 얼마나 존재하는지 확인한다."""
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
    """교점이 영상 주변 허용 범위 안에 있는지 확인한다."""
    margin_x = width * CORNER_MARGIN_RATIO
    margin_y = height * CORNER_MARGIN_RATIO
    return -margin_x <= point[0] <= width + margin_x and -margin_y <= point[1] <= height + margin_y

def build_fine_feature(top: LineCandidate, left: LineCandidate, right: LineCandidate, edges: np.ndarray) -> FineFeature | None:
    """고정된 TOP과 LEFT / RIGHT 후보의 교점으로 FINE feature를 만든다."""
    height, width = edges.shape
    tl = intersection(top.line_abc, left.line_abc)
    tr = intersection(top.line_abc, right.line_abc)
    if tl is None or tr is None:
        return None
    if not valid_corner(tl, width, height) or not valid_corner(tr, width, height) or tl[0] >= tr[0]:
        return None
    top_width = float(np.linalg.norm(tr - tl))
    if not width * MIN_TOP_WIDTH_RATIO <= top_width <= width * MAX_TOP_WIDTH_RATIO:
        return None
    center = 0.5 * (tl + tr)
    probe_y = min(height * 0.90, center[1] + height * SIDE_PROBE_HEIGHT_RATIO)
    left_x = line_x_at_y(left.line_abc, probe_y)
    right_x = line_x_at_y(right.line_abc, probe_y)
    if left_x is None or right_x is None:
        return None
    left_probe = np.asarray([left_x, probe_y], dtype=np.float64)
    right_probe = np.asarray([right_x, probe_y], dtype=np.float64)
    if left_probe[0] >= tl[0] or right_probe[0] <= tr[0] or left_probe[0] >= right_probe[0]:
        return None
    if edge_support(edges, tl, tr) < MIN_TOP_EDGE_SUPPORT:
        return None
    if edge_support(edges, tl, left_probe) < MIN_SIDE_EDGE_SUPPORT:
        return None
    if edge_support(edges, tr, right_probe) < MIN_SIDE_EDGE_SUPPORT:
        return None
    return FineFeature(
        angle_deg=normalize_angle(top.angle_deg),
        center_x_px=float(center[0]),
        center_y_px=float(center[1]),
        top_width_px=top_width,
        tl=tl,
        tr=tr,
        left_probe=left_probe,
        right_probe=right_probe,
    )

def detect_fine(image: np.ndarray) -> FineFeature | None:
    """공통 TOP 하나에 LEFT / RIGHT를 붙여 FINE feature를 만든다."""
    gray, edges = preprocess(image)
    top = detect_top(gray, edges)
    if top is None:
        return None
    left_candidates, right_candidates = detect_side_candidates(edges)
    valid_features = []
    for left in left_candidates:
        for right in right_candidates:
            feature = build_fine_feature(top, left, right, edges)
            if feature is not None:
                valid_features.append(feature)
    if not valid_features:
        return None
    return max(valid_features, key=lambda feature: feature.top_width_px)

# 화면 표시

def show_top(image: np.ndarray, top: LineCandidate | None, title: str) -> None:
    """TOP detector 결과를 표시한다."""
    output = image.copy()
    if top is not None:
        p1 = tuple(np.rint(top.segment[:2]).astype(int))
        p2 = tuple(np.rint(top.segment[2:]).astype(int))
        cv2.line(output, p1, p2, (0, 255, 255), 4, cv2.LINE_AA)
        text = f"{title}  angle={normalize_angle(top.angle_deg):+.2f}"
        cv2.putText(output, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Tote Align", output)
    cv2.waitKey(1)

def show_fine(image: np.ndarray, feature: FineFeature | None, title: str) -> None:
    """FINE detector 결과를 표시한다."""
    output = image.copy()
    if feature is not None:
        tl = tuple(np.rint(feature.tl).astype(int))
        tr = tuple(np.rint(feature.tr).astype(int))
        left_probe = tuple(np.rint(feature.left_probe).astype(int))
        right_probe = tuple(np.rint(feature.right_probe).astype(int))
        center = (int(round(feature.center_x_px)), int(round(feature.center_y_px)))
        cv2.line(output, tl, tr, (0, 255, 255), 4, cv2.LINE_AA)
        cv2.line(output, tl, left_probe, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.line(output, tr, right_probe, (255, 255, 0), 4, cv2.LINE_AA)
        cv2.circle(output, center, 8, (255, 0, 255), -1)
        text = f"{title}  angle={feature.angle_deg:+.2f}  cx={feature.center_x_px:.1f}  cy={feature.center_y_px:.1f}"
        cv2.putText(output, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Tote Align", output)
    cv2.waitKey(1)

# 측정

def measure_top(pipeline: rs.pipeline, title: str) -> TopMeasurement | None:
    """TOP을 여러 frame 측정하고 median 값을 사용한다."""
    features = []
    start_time = time.monotonic()
    while len(features) < MEASURE_FRAMES and time.monotonic() - start_time < MEASURE_TIMEOUT_S:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        image = np.asarray(color_frame.get_data())
        gray, edges = preprocess(image)
        top = detect_top(gray, edges)
        show_top(image, top, title)
        if top is not None:
            features.append(top)
    if len(features) < MEASURE_FRAMES // 2:
        print(f"TOP 측정 실패: {len(features)}/{MEASURE_FRAMES}")
        return None
    center_ys = np.asarray([feature.center_y for feature in features], dtype=np.float64)
    median_y = float(np.median(center_ys))
    filtered = [feature for feature in features if abs(feature.center_y - median_y) <= TOP_Y_FILTER_PX]
    if len(filtered) < MEASURE_FRAMES // 2:
        print("TOP 안정화 실패")
        return None
    angles = np.asarray([normalize_angle(feature.angle_deg) for feature in filtered], dtype=np.float64)
    center_xs = np.asarray([feature.center_x for feature in filtered], dtype=np.float64)
    center_ys = np.asarray([feature.center_y for feature in filtered], dtype=np.float64)
    measurement = TopMeasurement(
        angle_deg=float(np.median(angles)),
        center_x_px=float(np.median(center_xs)),
        center_y_px=float(np.median(center_ys)),
        angle_std_deg=float(np.std(angles)),
        center_y_std_px=float(np.std(center_ys)),
        valid_frames=len(filtered),
    )
    print(
        f"TOP 측정 | angle={measurement.angle_deg:+.3f} deg | cx={measurement.center_x_px:.1f} px | cy={measurement.center_y_px:.1f} px | "
        f"angle_std={measurement.angle_std_deg:.3f} | cy_std={measurement.center_y_std_px:.3f} | frames={measurement.valid_frames}"
    )
    return measurement

def measure_fine(pipeline: rs.pipeline, title: str) -> FineMeasurement | None:
    """FINE feature를 여러 frame 측정하고 median center_x / center_y를 사용한다."""
    features = []
    start_time = time.monotonic()
    while len(features) < MEASURE_FRAMES and time.monotonic() - start_time < MEASURE_TIMEOUT_S:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        image = np.asarray(color_frame.get_data())
        feature = detect_fine(image)
        show_fine(image, feature, title)
        if feature is not None:
            features.append(feature)
    if len(features) < MEASURE_FRAMES // 2:
        print(f"FINE 측정 실패: {len(features)}/{MEASURE_FRAMES}")
        return None
    center_ys = np.asarray([feature.center_y_px for feature in features], dtype=np.float64)
    widths = np.asarray([feature.top_width_px for feature in features], dtype=np.float64)
    median_y = float(np.median(center_ys))
    median_width = float(np.median(widths))
    filtered = [
        feature
        for feature in features
        if abs(feature.center_y_px - median_y) <= FINE_Y_FILTER_PX and abs(feature.top_width_px - median_width) <= FINE_WIDTH_FILTER_PX
    ]
    if len(filtered) < MEASURE_FRAMES // 2:
        print("FINE 안정화 실패")
        return None
    angles = np.asarray([feature.angle_deg for feature in filtered], dtype=np.float64)
    center_xs = np.asarray([feature.center_x_px for feature in filtered], dtype=np.float64)
    center_ys = np.asarray([feature.center_y_px for feature in filtered], dtype=np.float64)
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
        f"FINE 측정 | angle={measurement.angle_deg:+.3f} deg | cx={measurement.center_x_px:.2f} px | cy={measurement.center_y_px:.2f} px | "
        f"width={measurement.top_width_px:.2f} px | cx_std={measurement.center_x_std_px:.3f} | cy_std={measurement.center_y_std_px:.3f} | "
        f"frames={measurement.valid_frames}"
    )
    return measurement

# Base 이동

def flush_camera(pipeline: rs.pipeline) -> None:
    """로봇 이동 중 쌓인 이전 frame을 버린다."""
    for _ in range(CAMERA_FLUSH_FRAMES):
        pipeline.wait_for_frames()

def turn_duration(angle_rad: float) -> float:
    """Yaw trajectory 시간을 계산한다."""
    return max(QUINTIC_PEAK * abs(angle_rad) / ALIGN_ANGULAR_SPEED, MIN_LEG_TIME)

def linear_duration(distance_m: float) -> float:
    """Linear trajectory 시간을 계산한다."""
    return max(QUINTIC_PEAK * abs(distance_m) / ALIGN_LINEAR_SPEED, MIN_LEG_TIME)

def move_relative(robot, monitor: OdometryMonitor, x: float = 0.0, y: float = 0.0, yaw_deg: float = 0.0) -> bool:
    """현재 자세 기준으로 x / y / yaw 상대 이동을 수행한다."""
    yaw_rad = math.radians(yaw_deg)
    duration = turn_duration(yaw_rad) if abs(yaw_rad) > 1e-8 else linear_duration(max(abs(x), abs(y)))
    leg = build_leg(
        start=odom_pose(monitor.odom),
        target=(x, y, yaw_rad),
        absolute=False,
        duration=duration,
        turn_direction="shortest",
    )
    return move_leg(robot, monitor, leg, settle=SETTLE_S)

# Align

def align_yaw(robot, monitor: OdometryMonitor, pipeline: rs.pipeline) -> bool:
    """TOP angle을 한 번 측정하고 yaw를 한 번에 보정한다."""
    print()
    print("========== YAW ALIGN ==========")
    before = measure_top(pipeline, "YAW BEFORE")
    if before is None:
        return False
    error_deg = before.angle_deg - TARGET_ANGLE_DEG
    if abs(error_deg) <= ANGLE_TOL_DEG:
        print(f"이미 Yaw 정렬 범위 안입니다. error={abs(error_deg):.2f} deg")
        return True
    command_deg = YAW_SIGN * YAW_GAIN * error_deg
    command_deg = clamp(command_deg, -MAX_YAW_COMMAND_DEG, MAX_YAW_COMMAND_DEG)
    print(f"Yaw error   : {error_deg:+.3f} deg")
    print(f"Yaw command : {command_deg:+.3f} deg")
    if not move_relative(robot, monitor, yaw_deg=command_deg):
        return False
    time.sleep(SETTLE_S)
    flush_camera(pipeline)
    after = measure_top(pipeline, "YAW AFTER")
    if after is None:
        return False
    final_error_deg = after.angle_deg - TARGET_ANGLE_DEG
    print(f"Yaw before  : {before.angle_deg:+.3f} deg")
    print(f"Yaw after   : {after.angle_deg:+.3f} deg")
    print(f"Final error : {abs(final_error_deg):.2f} deg")
    return abs(final_error_deg) <= ANGLE_TOL_DEG

def align_lateral(robot, monitor: OdometryMonitor, pipeline: rs.pipeline) -> bool:
    """FINE center_x를 한 번 측정하고 lateral을 한 번에 보정한다."""
    print()
    print("========== LATERAL ALIGN ==========")
    before = measure_fine(pipeline, "LATERAL BEFORE")
    if before is None:
        return False
    error_px = before.center_x_px - TARGET_CENTER_X_PX
    error_m = abs(error_px * LATERAL_M_PER_PX)
    if error_m <= POSITION_TOL_M:
        print(f"이미 Lateral 정렬 범위 안입니다. error={error_m * 100:.2f} cm")
        return True
    command_m = (TARGET_CENTER_X_PX - before.center_x_px) * LATERAL_M_PER_PX
    command_m = clamp(command_m, -MAX_LATERAL_COMMAND_M, MAX_LATERAL_COMMAND_M)
    print(f"Center_x current : {before.center_x_px:.2f} px")
    print(f"Center_x target  : {TARGET_CENTER_X_PX:.2f} px")
    print(f"Estimated error  : {error_m * 100:.2f} cm")
    print(f"Lateral command  : {command_m:+.4f} m")
    if not move_relative(robot, monitor, y=command_m):
        return False
    time.sleep(SETTLE_S)
    flush_camera(pipeline)
    after = measure_fine(pipeline, "LATERAL AFTER")
    if after is None:
        return False
    final_error_px = after.center_x_px - TARGET_CENTER_X_PX
    final_error_m = abs(final_error_px * LATERAL_M_PER_PX)
    print(f"Lateral before : {before.center_x_px:.2f} px")
    print(f"Lateral after  : {after.center_x_px:.2f} px")
    print(f"Final error    : {final_error_m * 100:.2f} cm")
    return final_error_m <= POSITION_TOL_M

def align_forward(robot, monitor: OdometryMonitor, pipeline: rs.pipeline) -> bool:
    """FINE center_y를 한 번 측정하고 전후 위치를 한 번에 보정한다."""
    print()
    print("========== FORWARD ALIGN ==========")
    before = measure_fine(pipeline, "FORWARD BEFORE")
    if before is None:
        return False
    error_px = before.center_y_px - TARGET_CENTER_Y_PX
    error_m = abs(error_px * FORWARD_M_PER_PX)
    if error_m <= POSITION_TOL_M:
        print(f"이미 Forward 정렬 범위 안입니다. error={error_m * 100:.2f} cm")
        return True
    command_m = (TARGET_CENTER_Y_PX - before.center_y_px) * FORWARD_M_PER_PX
    command_m = clamp(command_m, -MAX_FORWARD_COMMAND_M, MAX_FORWARD_COMMAND_M)
    print(f"Center_y current : {before.center_y_px:.2f} px")
    print(f"Center_y target  : {TARGET_CENTER_Y_PX:.2f} px")
    print(f"Estimated error  : {error_m * 100:.2f} cm")
    print(f"Forward command  : {command_m:+.4f} m")
    if not move_relative(robot, monitor, x=command_m):
        return False
    time.sleep(SETTLE_S)
    flush_camera(pipeline)
    after = measure_fine(pipeline, "FORWARD AFTER")
    if after is None:
        return False
    final_error_px = after.center_y_px - TARGET_CENTER_Y_PX
    final_error_m = abs(final_error_px * FORWARD_M_PER_PX)
    print(f"Forward before : {before.center_y_px:.2f} px")
    print(f"Forward after  : {after.center_y_px:.2f} px")
    print(f"Final error    : {final_error_m * 100:.2f} cm")
    return final_error_m <= POSITION_TOL_M

# Main

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
    parser.add_argument("--serial", default=None)
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
        print("====================================")
        print("       TOTE VISION ALIGN")
        print("====================================")
        print(f"Target angle : {TARGET_ANGLE_DEG:+.2f} deg")
        print(f"Target cx    : {TARGET_CENTER_X_PX:.2f} px")
        print(f"Target cy    : {TARGET_CENTER_Y_PX:.2f} px")
        print(f"Position tol : ±{POSITION_TOL_M * 100:.1f} cm")
        if not align_yaw(robot, monitor, pipeline):
            print("\nRESULT: YAW FAILED")
            return
        time.sleep(SETTLE_S)
        flush_camera(pipeline)
        if not align_lateral(robot, monitor, pipeline):
            print("\nRESULT: LATERAL FAILED")
            return
        time.sleep(SETTLE_S)
        flush_camera(pipeline)
        if not align_forward(robot, monitor, pipeline):
            print("\nRESULT: FORWARD FAILED")
            return
        print()
        print("====================================")
        print("RESULT: SUCCESS")
        print("====================================")
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