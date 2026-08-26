#!/usr/bin/env python3
"""TOP rim과 왼쪽/오른쪽 위 모서리를 독립적으로 측정하는 calibration 전용 테스트.

목적
- LEFT feature  = [TL.x, TL.y, top_angle]
- RIGHT feature = [TR.x, TR.y, top_angle]
- 한쪽 모서리만 보여도 해당 feature를 계속 측정한다.
- 이후 x / y / yaw ± 이동 데이터를 이용해 one-shot SE(2) Jacobian을 계산한다.

키
- p: 최근 측정값 통계 출력
- r: 측정값 초기화
- s: 현재 화면 저장
- q / ESC: 종료

사용법:
python test/tote_corner_calibration_test.py --serial 250122079439
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


# 로봇 / 자동 Yaw calibration
ADDRESS = "192.168.30.1:50051"

SETTLE_S = 0.7
ALIGN_ANGULAR_SPEED = 0.5
QUINTIC_PEAK = 1.875
MIN_LEG_TIME = 1.5

YAW_TEST_DEG = 2.0
MEASURE_FRAMES = 40
MEASURE_TIMEOUT_S = 8.0
CAMERA_FLUSH_FRAMES = 10

# 카메라
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30
CAMERA_WARMUP_FRAMES = 30

# 공통 영상 처리
CANNY_LOW = 40
CANNY_HIGH = 120
HOUGH_THRESHOLD = 45
MAX_LINE_GAP = 40

# TOP rim
TOP_MIN_LINE_LENGTH = 100
TOP_MIN_Y = 55
TOP_MAX_Y = 175
TOP_MAX_ANGLE_DEG = 12.0

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

EDGE_SAMPLE_COUNT = 20
EDGE_SEARCH_RADIUS = 4
MIN_SIDE_EDGE_SUPPORT = 0.35

# 화면 / 통계
WINDOW_NAME = "Tote Corner Calibration"
MAX_HISTORY = 300


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
    """기존 V2와 같은 방식으로 TOP rim 하나를 선택한다."""
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

        if abs(angle_deg) > TOP_MAX_ANGLE_DEG or not TOP_MIN_Y <= candidate.center_y <= TOP_MAX_Y:
            continue

        length_score = candidate.length / CAM_WIDTH
        contrast_score = np.clip(top_contrast(gray, candidate.segment) / 80.0, -1.0, 1.0)
        score = 0.70 * length_score + 0.30 * contrast_score

        if score > best_score:
            best = candidate
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

    return FrameFeature(
        top=top,
        top_angle_deg=normalize_angle(top.angle_deg),
        left=choose_left_corner(top, left_candidates, edges),
        right=choose_right_corner(top, right_candidates, edges),
    )


def draw_feature(image: np.ndarray, feature: FrameFeature | None) -> np.ndarray:
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

    return output


def print_array_stats(name: str, values: list[float], unit: str) -> None:
    """한 항목의 mean / median / std / min / max를 출력한다."""
    if not values:
        print(f"{name:14s}: no data")
        return

    array = np.asarray(values, dtype=np.float64)
    print(
        f"{name:14s}: mean={np.mean(array):+9.3f} | median={np.median(array):+9.3f} | "
        f"std={np.std(array):7.3f} | min={np.min(array):+9.3f} | max={np.max(array):+9.3f} {unit}"
    )


def print_stats(history: list[FrameFeature]) -> None:
    """최근 history에서 LEFT / RIGHT feature 통계를 각각 출력한다."""
    if not history:
        print("측정값이 없습니다.")
        return

    angles = [feature.top_angle_deg for feature in history]
    left = [feature for feature in history if feature.left is not None]
    right = [feature for feature in history if feature.right is not None]
    full = [feature for feature in history if feature.left is not None and feature.right is not None]

    print()
    print("=" * 96)
    print(f"최근 측정 {len(history)} frames | TOP={len(history)} | LEFT={len(left)} | RIGHT={len(right)} | BOTH={len(full)}")
    print("-" * 96)
    print_array_stats("top_angle", angles, "deg")

    print()
    print("[LEFT feature = TL.x, TL.y, top_angle]")
    print_array_stats("TL.x", [feature.left.point[0] for feature in left], "px")
    print_array_stats("TL.y", [feature.left.point[1] for feature in left], "px")
    print_array_stats("LEFT angle", [feature.top_angle_deg for feature in left], "deg")

    print()
    print("[RIGHT feature = TR.x, TR.y, top_angle]")
    print_array_stats("TR.x", [feature.right.point[0] for feature in right], "px")
    print_array_stats("TR.y", [feature.right.point[1] for feature in right], "px")
    print_array_stats("RIGHT angle", [feature.top_angle_deg for feature in right], "deg")

    if full:
        widths = [float(np.linalg.norm(feature.right.point - feature.left.point)) for feature in full]
        print()
        print_array_stats("TOP width", widths, "px")

    print("=" * 96)
    print()


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



@dataclass
class CalibrationMeasurement:
    """한 자세에서 여러 frame을 모아 계산한 median feature."""

    angle_deg: float
    tl_x: float | None
    tl_y: float | None
    tr_x: float | None
    tr_y: float | None
    top_frames: int
    left_frames: int
    right_frames: int


def wrap_angle_rad(angle: float) -> float:
    """각도를 -pi~pi 범위로 정규화한다."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def turn_duration(angle_rad: float) -> float:
    """Yaw trajectory 시간을 계산한다."""
    return max(QUINTIC_PEAK * abs(angle_rad) / ALIGN_ANGULAR_SPEED, MIN_LEG_TIME)


def move_relative_yaw(robot, monitor: OdometryMonitor, angle_deg: float) -> bool:
    """현재 odom 자세 기준으로 yaw를 상대 이동한다."""
    angle_rad = math.radians(angle_deg)

    leg = build_leg(
        start=odom_pose(monitor.odom),
        target=(0.0, 0.0, angle_rad),
        absolute=False,
        duration=turn_duration(angle_rad),
        turn_direction="shortest",
    )

    return move_leg(robot, monitor, leg, settle=SETTLE_S)


def flush_camera(pipeline: rs.pipeline) -> None:
    """로봇 이동 중 쌓인 이전 frame을 버린다."""
    for _ in range(CAMERA_FLUSH_FRAMES):
        pipeline.wait_for_frames()


def measure_pose(pipeline: rs.pipeline, label: str, frame_count: int) -> CalibrationMeasurement | None:
    """현재 자세에서 TOP / TL / TR을 여러 frame 측정하고 median을 반환한다."""
    angles: list[float] = []
    tl_xs: list[float] = []
    tl_ys: list[float] = []
    tr_xs: list[float] = []
    tr_ys: list[float] = []

    start_time = time.monotonic()

    while len(angles) < frame_count and time.monotonic() - start_time < MEASURE_TIMEOUT_S:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asarray(color_frame.get_data())
        feature = detect_frame_feature(image)
        output = draw_feature(image, feature)

        cv2.putText(output, label, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, output)
        cv2.waitKey(1)

        if feature is None:
            continue

        angles.append(feature.top_angle_deg)

        if feature.left is not None:
            tl_xs.append(float(feature.left.point[0]))
            tl_ys.append(float(feature.left.point[1]))

        if feature.right is not None:
            tr_xs.append(float(feature.right.point[0]))
            tr_ys.append(float(feature.right.point[1]))

    if len(angles) < frame_count // 2:
        print(f"{label}: TOP 측정 실패 ({len(angles)}/{frame_count})")
        return None

    measurement = CalibrationMeasurement(
        angle_deg=float(np.median(angles)),
        tl_x=float(np.median(tl_xs)) if tl_xs else None,
        tl_y=float(np.median(tl_ys)) if tl_ys else None,
        tr_x=float(np.median(tr_xs)) if tr_xs else None,
        tr_y=float(np.median(tr_ys)) if tr_ys else None,
        top_frames=len(angles),
        left_frames=len(tl_xs),
        right_frames=len(tr_xs),
    )

    print_measurement(label, measurement)
    return measurement


def print_measurement(label: str, measurement: CalibrationMeasurement) -> None:
    """한 자세의 calibration 측정값을 출력한다."""
    tl_text = "N/A" if measurement.tl_x is None else f"({measurement.tl_x:.3f}, {measurement.tl_y:.3f})"
    tr_text = "N/A" if measurement.tr_x is None else f"({measurement.tr_x:.3f}, {measurement.tr_y:.3f})"

    print()
    print(f"[{label}]")
    print(f"TOP angle : {measurement.angle_deg:+.4f} deg")
    print(f"TL        : {tl_text}   frames={measurement.left_frames}")
    print(f"TR        : {tr_text}   frames={measurement.right_frames}")
    print(f"TOP frames: {measurement.top_frames}")


def odom_yaw_delta_deg(monitor: OdometryMonitor, reference_yaw_rad: float) -> float:
    """초기 기준 heading으로부터 현재 odom yaw 차이를 degree로 반환한다."""
    current_yaw_rad = odom_pose(monitor.odom)[2]
    return math.degrees(wrap_angle_rad(current_yaw_rad - reference_yaw_rad))


def print_calibration_result(
    reference: CalibrationMeasurement,
    plus: CalibrationMeasurement,
    minus: CalibrationMeasurement,
    plus_yaw_deg: float,
    minus_yaw_deg: float,
    final_yaw_deg: float,
) -> None:
    """+/- Yaw 측정값으로 image-angle gain과 corner yaw column을 계산한다."""
    robot_span_deg = plus_yaw_deg - minus_yaw_deg
    image_span_deg = plus.angle_deg - minus.angle_deg

    print()
    print("=" * 94)
    print("AUTO YAW CALIBRATION RESULT")
    print("=" * 94)
    print(f"Reference image angle : {reference.angle_deg:+.4f} deg")
    print(f"+Yaw actual odom      : {plus_yaw_deg:+.4f} deg")
    print(f"-Yaw actual odom      : {minus_yaw_deg:+.4f} deg")
    print(f"Final return odom     : {final_yaw_deg:+.4f} deg")
    print(f"Robot yaw span        : {robot_span_deg:+.4f} deg")
    print(f"Image angle span      : {image_span_deg:+.4f} deg")

    if abs(robot_span_deg) < 1e-6 or abs(image_span_deg) < 1e-6:
        print("Yaw span이 너무 작아서 gain을 계산할 수 없습니다.")
        print("=" * 94)
        return

    image_per_robot = image_span_deg / robot_span_deg
    correction_per_image = -robot_span_deg / image_span_deg

    print()
    print(f"d(image_angle) / d(robot_yaw) = {image_per_robot:+.6f} deg/deg")
    print(f"yaw_command / image_error      = {correction_per_image:+.6f} deg/deg")
    print()
    print("현재 align 코드 형태로 쓰면:")
    print(f"TARGET_ANGLE_DEG = {reference.angle_deg:+.6f}")
    print(f"YAW_SIGN = {1.0 if correction_per_image >= 0.0 else -1.0:+.1f}")
    print(f"YAW_GAIN = {abs(correction_per_image):.6f}")

    if plus.tl_x is not None and minus.tl_x is not None:
        d_tlx = (plus.tl_x - minus.tl_x) / robot_span_deg
        d_tly = (plus.tl_y - minus.tl_y) / robot_span_deg
        d_angle = image_span_deg / robot_span_deg

        print()
        print("LEFT feature yaw column  [TL.x, TL.y, top_angle] / robot yaw deg")
        print(f"J_LEFT_YAW = [{d_tlx:+.6f}, {d_tly:+.6f}, {d_angle:+.6f}]")

    if plus.tr_x is not None and minus.tr_x is not None:
        d_trx = (plus.tr_x - minus.tr_x) / robot_span_deg
        d_try = (plus.tr_y - minus.tr_y) / robot_span_deg
        d_angle = image_span_deg / robot_span_deg

        print()
        print("RIGHT feature yaw column [TR.x, TR.y, top_angle] / robot yaw deg")
        print(f"J_RIGHT_YAW = [{d_trx:+.6f}, {d_try:+.6f}, {d_angle:+.6f}]")

    print("=" * 94)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=None, help="사용할 D435 serial")
    parser.add_argument("--yaw-deg", type=float, default=YAW_TEST_DEG, help="자동으로 회전할 yaw 크기")
    parser.add_argument("--frames", type=int, default=MEASURE_FRAMES, help="각 자세에서 사용할 TOP frame 수")
    parser.add_argument("--address", default=ADDRESS, help="RBY1 주소")
    args = parser.parse_args()

    pipeline = start_camera(args.serial)
    robot = initialize_mobile(address=args.address, model="m")
    monitor = OdometryMonitor()

    try:
        robot.start_state_update(monitor.on_state, rate=50)

        if not wait_for_odometry(monitor):
            print("Odometry를 받지 못했습니다.")
            return

        print()
        print("==============================================================")
        print("        TOTE AUTO YAW CALIBRATION")
        print("==============================================================")
        print(f"Yaw command : ±{args.yaw_deg:.2f} deg")
        print(f"Frames      : {args.frames}")
        print()
        print("로봇을 grasp 성공 기준 자세에 놓아주세요.")
        print("Enter를 누르면 REFERENCE -> +Yaw -> 기준 복귀 -> -Yaw -> 기준 복귀 순서로 자동 진행합니다.")
        input("준비되면 Enter > ")

        flush_camera(pipeline)

        reference = measure_pose(pipeline, "REFERENCE", args.frames)

        if reference is None:
            return

        reference_yaw_rad = odom_pose(monitor.odom)[2]

        print(f"\n[1/4] +{args.yaw_deg:.2f} deg 회전")
        if not move_relative_yaw(robot, monitor, +args.yaw_deg):
            print("+Yaw 이동 실패")
            return

        flush_camera(pipeline)
        plus_yaw_deg = odom_yaw_delta_deg(monitor, reference_yaw_rad)
        plus = measure_pose(pipeline, "+YAW", args.frames)

        if plus is None:
            return

        print(f"\n[2/4] 기준 자세로 복귀: {-args.yaw_deg:.2f} deg")
        if not move_relative_yaw(robot, monitor, -args.yaw_deg):
            print("기준 복귀 실패")
            return

        flush_camera(pipeline)
        return1_yaw_deg = odom_yaw_delta_deg(monitor, reference_yaw_rad)
        print(f"기준 복귀 odom yaw error: {return1_yaw_deg:+.4f} deg")

        print(f"\n[3/4] -{args.yaw_deg:.2f} deg 회전")
        if not move_relative_yaw(robot, monitor, -args.yaw_deg):
            print("-Yaw 이동 실패")
            return

        flush_camera(pipeline)
        minus_yaw_deg = odom_yaw_delta_deg(monitor, reference_yaw_rad)
        minus = measure_pose(pipeline, "-YAW", args.frames)

        if minus is None:
            return

        print(f"\n[4/4] 기준 자세로 복귀: +{args.yaw_deg:.2f} deg")
        if not move_relative_yaw(robot, monitor, +args.yaw_deg):
            print("최종 복귀 실패")
            return

        flush_camera(pipeline)
        final_yaw_deg = odom_yaw_delta_deg(monitor, reference_yaw_rad)
        final_reference = measure_pose(pipeline, "FINAL REFERENCE", args.frames)

        if final_reference is not None:
            print(f"Reference angle drift : {final_reference.angle_deg - reference.angle_deg:+.4f} deg")

        print_calibration_result(reference, plus, minus, plus_yaw_deg, minus_yaw_deg, final_yaw_deg)

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
