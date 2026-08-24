#!/usr/bin/env python3
"""D435 RGB에서 토트 opening의 사다리꼴과 4개 corner를 검출한다.

Canny + HoughLinesP로 rim 후보선을 찾고, 네 선 조합의 기하 구조와 내부 밝기, edge support를 평가한다.
Depth와 로봇 제어는 사용하지 않으며, 현재 단계에서는 TL/TR/BR/BL corner 검출 안정성만 확인한다.

사용법:
python test/tote_rim_test.py --serial 250122079439
"""

from __future__ import annotations

import argparse
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
MAX_LINE_GAP = 35


# ============================================================
# 관심 영역
# ============================================================

ROI_LEFT_RATIO = 0.01
ROI_RIGHT_RATIO = 0.99
ROI_TOP_RATIO = 0.10
ROI_BOTTOM_RATIO = 0.99


# ============================================================
# 후보선 각도
# ============================================================

HORIZONTAL_MAX_DEG = 18.0

LEFT_MIN_DEG = 92.0
LEFT_MAX_DEG = 140.0

RIGHT_MIN_DEG = 40.0
RIGHT_MAX_DEG = 88.0

MAX_HORIZONTAL_CANDIDATES = 14
MAX_LEFT_CANDIDATES = 10
MAX_RIGHT_CANDIDATES = 10


# ============================================================
# 사다리꼴 조건
# ============================================================

MIN_OPENING_HEIGHT_RATIO = 0.22
MIN_OPENING_WIDTH_RATIO = 0.40
MIN_AREA_RATIO = 0.16

MIN_BACK_FRONT_WIDTH_RATIO = 0.45
MAX_BACK_FRONT_WIDTH_RATIO = 1.05

CORNER_MARGIN_RATIO = 0.08


# ============================================================
# 밝기 / Edge score
# ============================================================

DARK_THRESHOLD = 100
MIN_DARK_FRACTION = 0.45

EDGE_SAMPLE_COUNT = 50
EDGE_SEARCH_RADIUS = 3


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
class ToteDetection:
    """최종 선택된 토트 opening."""

    corners: np.ndarray
    score: float
    dark_fraction: float
    edge_support: float


def line_info(segment: np.ndarray):
    """선분의 길이, 각도, 중심점을 계산한다."""
    x1, y1, x2, y2 = [float(value) for value in segment]

    dx = x2 - x1
    dy = y2 - y1

    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    return length, angle, center_x, center_y


def segment_to_line(segment: np.ndarray) -> np.ndarray:
    """두 점 선분을 ax + by + c = 0 형태의 무한 직선으로 변환한다."""
    x1, y1, x2, y2 = [float(value) for value in segment]

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = math.hypot(a, b)

    if norm < 1e-9:
        raise ValueError("길이가 0인 선분입니다.")

    return np.asarray([a / norm, b / norm, c / norm], dtype=np.float64)


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


def is_horizontal(angle_deg: float) -> bool:
    """0도 또는 180도 근처의 선인지 확인한다."""
    return angle_deg <= HORIZONTAL_MAX_DEG or angle_deg >= 180.0 - HORIZONTAL_MAX_DEG


def make_candidate(segment: np.ndarray) -> LineCandidate:
    """Hough 선분을 후보 객체로 변환한다."""
    length, angle, center_x, center_y = line_info(segment)

    return LineCandidate(
        segment=np.asarray(segment, dtype=np.float64),
        length=length,
        angle_deg=angle,
        center_x=center_x,
        center_y=center_y,
        line_abc=segment_to_line(segment),
    )


def detect_line_candidates(image: np.ndarray):
    """Hough 후보를 horizontal / left / right 세 그룹으로 나눈다."""
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    x0 = int(width * ROI_LEFT_RATIO)
    x1 = int(width * ROI_RIGHT_RATIO)
    y0 = int(height * ROI_TOP_RATIO)
    y1 = int(height * ROI_BOTTOM_RATIO)

    roi_mask = np.zeros_like(edges)
    roi_mask[y0:y1, x0:x1] = 255
    roi_edges = cv2.bitwise_and(edges, roi_mask)

    detected = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    if detected is None:
        return [], [], [], gray, roi_edges

    segments = np.asarray(detected, dtype=np.int32).reshape(-1, 4)

    horizontal = []
    left = []
    right = []

    for segment in segments:
        candidate = make_candidate(segment)

        if candidate.length < MIN_LINE_LENGTH:
            continue

        angle = candidate.angle_deg

        if is_horizontal(angle):
            horizontal.append(candidate)
        elif LEFT_MIN_DEG <= angle <= LEFT_MAX_DEG and candidate.center_x < width * 0.55:
            left.append(candidate)
        elif RIGHT_MIN_DEG <= angle <= RIGHT_MAX_DEG and candidate.center_x > width * 0.45:
            right.append(candidate)

    horizontal.sort(key=lambda item: item.length, reverse=True)
    left.sort(key=lambda item: item.length, reverse=True)
    right.sort(key=lambda item: item.length, reverse=True)

    horizontal = horizontal[:MAX_HORIZONTAL_CANDIDATES]
    left = left[:MAX_LEFT_CANDIDATES]
    right = right[:MAX_RIGHT_CANDIDATES]

    return horizontal, left, right, gray, roi_edges


def polygon_area(corners: np.ndarray) -> float:
    """4점 polygon의 면적을 계산한다."""
    return abs(float(cv2.contourArea(corners.astype(np.float32))))


def corners_inside_margin(corners: np.ndarray, width: int, height: int) -> bool:
    """교점이 화면에서 지나치게 멀리 벗어나지 않는지 확인한다."""
    margin_x = width * CORNER_MARGIN_RATIO
    margin_y = height * CORNER_MARGIN_RATIO

    x = corners[:, 0]
    y = corners[:, 1]

    return bool(
        np.all(x >= -margin_x)
        and np.all(x <= width + margin_x)
        and np.all(y >= -margin_y)
        and np.all(y <= height + margin_y)
    )


def geometry_score(corners: np.ndarray, width: int, height: int) -> float | None:
    """TL, TR, BR, BL이 정상적인 토트 사다리꼴인지 검사한다."""
    tl, tr, br, bl = corners

    if not corners_inside_margin(corners, width, height):
        return None

    if not (tl[0] < tr[0] and bl[0] < br[0]):
        return None

    if not (tl[1] < bl[1] and tr[1] < br[1]):
        return None

    back_width = float(np.linalg.norm(tr - tl))
    front_width = float(np.linalg.norm(br - bl))
    left_height = float(np.linalg.norm(bl - tl))
    right_height = float(np.linalg.norm(br - tr))
    mean_height = 0.5 * (left_height + right_height)

    if mean_height < height * MIN_OPENING_HEIGHT_RATIO:
        return None

    if min(back_width, front_width) < width * MIN_OPENING_WIDTH_RATIO:
        return None

    width_ratio = back_width / max(front_width, 1e-6)

    if not MIN_BACK_FRONT_WIDTH_RATIO <= width_ratio <= MAX_BACK_FRONT_WIDTH_RATIO:
        return None

    # 원근상 위쪽 모서리는 아래쪽 모서리보다 안쪽으로 들어오는 형태를 기대한다.
    if tl[0] < bl[0] - width * 0.05:
        return None

    if tr[0] > br[0] + width * 0.05:
        return None

    area = polygon_area(corners)

    if area < width * height * MIN_AREA_RATIO:
        return None

    height_ratio = min(left_height, right_height) / max(left_height, right_height, 1e-6)

    if height_ratio < 0.45:
        return None

    area_score = min(1.0, area / (width * height * 0.55))
    symmetry_score = height_ratio
    perspective_score = 1.0 - min(abs(width_ratio - 0.85), 0.85) / 0.85

    return 0.45 * area_score + 0.30 * symmetry_score + 0.25 * perspective_score


def polygon_dark_fraction(gray: np.ndarray, corners: np.ndarray) -> float:
    """사다리꼴 내부가 얼마나 어두운지 계산한다."""
    mask = np.zeros_like(gray, dtype=np.uint8)
    polygon = np.rint(corners).astype(np.int32)

    cv2.fillConvexPoly(mask, polygon, 255)

    # rim 자체보다 실제 opening 내부를 보기 위해 mask를 조금 줄인다.
    kernel = np.ones((15, 15), dtype=np.uint8)
    inner_mask = cv2.erode(mask, kernel, iterations=1)

    pixels = gray[inner_mask > 0]

    if pixels.size == 0:
        return 0.0

    return float(np.mean(pixels < DARK_THRESHOLD))


def segment_edge_support(edges: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """두 corner 사이에 실제 edge가 얼마나 이어져 있는지 계산한다."""
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

        patch = edges[y0:y1, x0:x1]

        valid += 1

        if np.any(patch > 0):
            supported += 1

    if valid == 0:
        return 0.0

    return supported / valid


def trapezoid_edge_support(edges: np.ndarray, corners: np.ndarray) -> float:
    """BACK / RIGHT / FRONT / LEFT 네 변의 평균 edge support를 계산한다."""
    tl, tr, br, bl = corners

    supports = [
        segment_edge_support(edges, tl, tr),
        segment_edge_support(edges, tr, br),
        segment_edge_support(edges, br, bl),
        segment_edge_support(edges, bl, tl),
    ]

    return float(np.mean(supports))


def make_corners(
    back: LineCandidate,
    front: LineCandidate,
    left: LineCandidate,
    right: LineCandidate,
) -> np.ndarray | None:
    """네 직선의 교점으로 TL, TR, BR, BL을 만든다."""
    tl = intersection(back.line_abc, left.line_abc)
    tr = intersection(back.line_abc, right.line_abc)
    bl = intersection(front.line_abc, left.line_abc)
    br = intersection(front.line_abc, right.line_abc)

    if any(point is None for point in (tl, tr, br, bl)):
        return None

    return np.stack((tl, tr, br, bl), axis=0)


def find_best_tote(image: np.ndarray) -> tuple[ToteDetection | None, np.ndarray]:
    """모든 후보선 조합 중 가장 그럴듯한 토트 opening 하나를 선택한다."""
    horizontal, left_candidates, right_candidates, gray, edges = detect_line_candidates(image)

    height, width = image.shape[:2]
    best_detection = None

    for first_index in range(len(horizontal)):
        for second_index in range(first_index + 1, len(horizontal)):
            first = horizontal[first_index]
            second = horizontal[second_index]

            back, front = (first, second) if first.center_y <= second.center_y else (second, first)

            # 같은 rim의 안쪽/바깥쪽 edge를 BACK과 FRONT로 동시에 선택하지 못하게 한다.
            if front.center_y - back.center_y < height * MIN_OPENING_HEIGHT_RATIO:
                continue

            for left in left_candidates:
                for right in right_candidates:
                    corners = make_corners(back, front, left, right)

                    if corners is None:
                        continue

                    geo_score = geometry_score(corners, width, height)

                    if geo_score is None:
                        continue

                    dark_fraction = polygon_dark_fraction(gray, corners)

                    if dark_fraction < MIN_DARK_FRACTION:
                        continue

                    edge_support = trapezoid_edge_support(edges, corners)

                    if edge_support < 0.35:
                        continue

                    total_score = 0.35 * geo_score + 0.35 * dark_fraction + 0.30 * edge_support

                    if best_detection is None or total_score > best_detection.score:
                        best_detection = ToteDetection(
                            corners=corners,
                            score=total_score,
                            dark_fraction=dark_fraction,
                            edge_support=edge_support,
                        )

    return best_detection, edges


def draw_detection(image: np.ndarray, detection: ToteDetection | None) -> np.ndarray:
    """최종 토트 opening과 TL/TR/BR/BL corner를 영상에 표시한다."""
    output = image.copy()

    if detection is None:
        cv2.putText(output, "TOTE NOT FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return output

    corners = np.rint(detection.corners).astype(np.int32)
    tl, tr, br, bl = corners

    cv2.line(output, tuple(tl), tuple(tr), (0, 255, 255), 4, cv2.LINE_AA)
    cv2.line(output, tuple(tr), tuple(br), (255, 255, 0), 4, cv2.LINE_AA)
    cv2.line(output, tuple(bl), tuple(br), (255, 0, 255), 4, cv2.LINE_AA)
    cv2.line(output, tuple(tl), tuple(bl), (0, 255, 0), 4, cv2.LINE_AA)

    for name, point in zip(("TL", "TR", "BR", "BL"), corners):
        point_tuple = (int(point[0]), int(point[1]))

        cv2.circle(output, point_tuple, 7, (0, 0, 255), -1)
        cv2.putText(output, name, (point_tuple[0] + 8, point_tuple[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    center = np.mean(detection.corners, axis=0)
    center_point = (int(round(center[0])), int(round(center[1])))

    cv2.circle(output, center_point, 8, (255, 255, 255), -1)
    cv2.putText(output, "TOTE FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    score_text = f"score={detection.score:.3f}  dark={detection.dark_fraction:.2f}  edge={detection.edge_support:.2f}"
    cv2.putText(output, score_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    return output


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

    print("q 또는 ESC: 종료")
    print("s: 현재 결과 저장")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            image = np.asarray(color_frame.get_data())
            detection, edges = find_best_tote(image)
            result = draw_detection(image, detection)

            cv2.imshow("Tote Opening", result)
            cv2.imshow("Edges", edges)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            if key in (ord("s"), ord("S")):
                cv2.imwrite("tote_rgb.png", image)
                cv2.imwrite("tote_detection.png", result)
                cv2.imwrite("tote_edges.png", edges)

                print("tote_rgb.png / tote_detection.png / tote_edges.png 저장")

    finally:
        pipeline.stop()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()