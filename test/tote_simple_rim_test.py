#!/usr/bin/env python3
"""D435 RGB에서 토트 위쪽 rim의 영상 feature를 측정한다.

출력 feature
- angle_deg: 수평 기준 rim 각도
- center_x_px: rim 중심 x
- center_y_px: rim 중심 y
- width_px: 영상에서 보이는 rim 길이

로봇은 움직이지 않는다.

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
# 위쪽 rim 검출
# ============================================================

CANNY_LOW = 40
CANNY_HIGH = 120

HOUGH_THRESHOLD = 50
MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 40

# 위쪽 rim이 존재하는 화면 영역만 사용한다.
ROI_LEFT_RATIO = 0.03
ROI_RIGHT_RATIO = 0.97
ROI_TOP_RATIO = 0.15
ROI_BOTTOM_RATIO = 0.60

# 수평에서 이 각도 이상 기울어진 선은 제외한다.
MAX_HORIZONTAL_ANGLE_DEG = 15.0

# 같은 rim으로 묶을 Hough 선들의 허용 차이.
RIM_GROUP_Y_TOL_PX = 15.0
RIM_GROUP_ANGLE_TOL_DEG = 4.0

# 최근 N개 유효 측정값을 통계에 사용한다.
HISTORY_SIZE = 60

# 터미널 출력 간격.
PRINT_EVERY_N_FRAMES = 5


@dataclass
class ToteFeature:
    """위쪽 rim에서 얻은 영상 feature."""

    angle_deg: float
    center_x_px: float
    center_y_px: float
    width_px: float
    line_p1: tuple[int, int]
    line_p2: tuple[int, int]


def normalize_horizontal_angle_deg(angle_deg: float) -> float:
    """line angle을 수평 기준 -90~90 deg 범위로 정규화한다."""
    angle = (float(angle_deg) + 180.0) % 180.0

    if angle >= 90.0:
        angle -= 180.0

    return angle


def line_angle_difference_deg(first: float, second: float) -> float:
    """180도 대칭 line angle 두 개의 최소 차이를 계산한다."""
    difference = normalize_horizontal_angle_deg(first - second)
    return abs(difference)


def line_info(line: np.ndarray):
    """Hough 선분의 길이, 수평 기준 각도, 중심점을 계산한다."""
    x1, y1, x2, y2 = [float(value) for value in line]

    dx = x2 - x1
    dy = y2 - y1

    length = math.hypot(dx, dy)
    angle_deg = normalize_horizontal_angle_deg(math.degrees(math.atan2(dy, dx)))
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    return length, angle_deg, center_x, center_y


def fitted_y_at_x(vx: float, vy: float, x0: float, y0: float, x: float) -> float:
    """cv2.fitLine 직선에서 지정 x 위치의 y를 계산한다."""
    if abs(vx) < 1e-8:
        return float(y0)

    return float(y0 + (x - x0) * vy / vx)


def detect_top_rim(image: np.ndarray) -> tuple[ToteFeature | None, np.ndarray]:
    """위쪽 rim에 속하는 Hough 선들을 묶고 하나의 대표 직선을 계산한다."""
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
        return None, roi_edges

    lines = np.asarray(detected, dtype=np.int32).reshape(-1, 4)
    candidates = []

    for line in lines:
        length, angle_deg, center_x, center_y = line_info(line)

        if length < MIN_LINE_LENGTH:
            continue

        if abs(angle_deg) > MAX_HORIZONTAL_ANGLE_DEG:
            continue

        candidates.append((length, angle_deg, center_x, center_y, line))

    if not candidates:
        return None, roi_edges

    # 가장 긴 수평선을 anchor로 잡고 주변의 같은 rim edge들을 함께 사용한다.
    candidates.sort(key=lambda item: item[0], reverse=True)

    anchor_length, anchor_angle, _, anchor_y, _ = candidates[0]
    grouped_lines = []

    for length, angle_deg, center_x, center_y, line in candidates:
        if abs(center_y - anchor_y) > RIM_GROUP_Y_TOL_PX:
            continue

        if line_angle_difference_deg(angle_deg, anchor_angle) > RIM_GROUP_ANGLE_TOL_DEG:
            continue

        grouped_lines.append(line)

    if not grouped_lines:
        return None, roi_edges

    # 같은 rim에 속한 선분들의 모든 endpoint를 모아 하나의 대표 직선을 fitting한다.
    points = []

    for line in grouped_lines:
        x_start, y_start, x_end, y_end = line
        points.append((x_start, y_start))
        points.append((x_end, y_end))

    points = np.asarray(points, dtype=np.float32)

    vx, vy, fit_x, fit_y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)

    # 방향을 항상 영상의 왼쪽 → 오른쪽으로 통일한다.
    if vx < 0.0:
        vx = -vx
        vy = -vy

    angle_deg = normalize_horizontal_angle_deg(math.degrees(math.atan2(float(vy), float(vx))))

    # Hough가 동일 rim에서 실제로 관측한 가장 왼쪽/오른쪽 endpoint를 사용한다.
    left_x = float(np.min(points[:, 0]))
    right_x = float(np.max(points[:, 0]))

    left_y = fitted_y_at_x(float(vx), float(vy), float(fit_x), float(fit_y), left_x)
    right_y = fitted_y_at_x(float(vx), float(vy), float(fit_x), float(fit_y), right_x)

    center_x = 0.5 * (left_x + right_x)
    center_y = fitted_y_at_x(float(vx), float(vy), float(fit_x), float(fit_y), center_x)
    width_px = math.hypot(right_x - left_x, right_y - left_y)

    feature = ToteFeature(
        angle_deg=angle_deg,
        center_x_px=center_x,
        center_y_px=center_y,
        width_px=width_px,
        line_p1=(int(round(left_x)), int(round(left_y))),
        line_p2=(int(round(right_x)), int(round(right_y))),
    )

    return feature, roi_edges


def draw_feature(image: np.ndarray, feature: ToteFeature | None) -> np.ndarray:
    """검출된 top rim과 네 feature 값을 영상에 표시한다."""
    output = image.copy()
    height, width = output.shape[:2]

    roi_x0 = int(width * ROI_LEFT_RATIO)
    roi_x1 = int(width * ROI_RIGHT_RATIO)
    roi_y0 = int(height * ROI_TOP_RATIO)
    roi_y1 = int(height * ROI_BOTTOM_RATIO)

    cv2.rectangle(output, (roi_x0, roi_y0), (roi_x1, roi_y1), (150, 150, 150), 1)

    if feature is None:
        cv2.putText(output, "TOP RIM NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return output

    cv2.line(output, feature.line_p1, feature.line_p2, (0, 255, 255), 4, cv2.LINE_AA)

    center = (int(round(feature.center_x_px)), int(round(feature.center_y_px)))
    cv2.circle(output, center, 8, (0, 0, 255), -1)

    # 영상 정중앙도 함께 표시해 reference와 비교하기 쉽게 한다.
    image_center = (width // 2, height // 2)
    cv2.drawMarker(output, image_center, (255, 0, 255), cv2.MARKER_CROSS, 22, 2)

    cv2.putText(output, "TOP RIM FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    text = (
        f"angle={feature.angle_deg:+.2f} deg  "
        f"cx={feature.center_x_px:.1f}  cy={feature.center_y_px:.1f}  width={feature.width_px:.1f}"
    )

    cv2.putText(output, text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA)

    return output


def print_feature(feature: ToteFeature) -> None:
    """한 프레임의 feature를 터미널에 출력한다."""
    print(
        f"angle={feature.angle_deg:+7.3f} deg | "
        f"center_x={feature.center_x_px:7.2f} px | "
        f"center_y={feature.center_y_px:7.2f} px | "
        f"width={feature.width_px:7.2f} px"
    )


def print_summary(history: deque[ToteFeature]) -> None:
    """최근 유효 frame의 평균, 표준편차, 범위를 출력한다."""
    if not history:
        print("측정값이 없습니다.")
        return

    values = np.asarray(
        [[item.angle_deg, item.center_x_px, item.center_y_px, item.width_px] for item in history],
        dtype=np.float64,
    )

    names = ("angle_deg", "center_x_px", "center_y_px", "width_px")
    units = ("deg", "px", "px", "px")

    print()
    print("=" * 72)
    print(f"최근 유효 측정 {len(history)} frames")

    for index, (name, unit) in enumerate(zip(names, units)):
        mean = float(np.mean(values[:, index]))
        std = float(np.std(values[:, index]))
        minimum = float(np.min(values[:, index]))
        maximum = float(np.max(values[:, index]))

        print(f"{name:12s}: mean={mean:+9.3f} {unit} | std={std:7.3f} | min={minimum:+9.3f} | max={maximum:+9.3f}")

    print("=" * 72)
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
            feature, _ = detect_top_rim(image)

            if feature is not None:
                history.append(feature)

                if frame_count % PRINT_EVERY_N_FRAMES == 0:
                    print_feature(feature)

            result = draw_feature(image, feature)
            last_result = result

            cv2.imshow("Tote Top Rim Feature", result)

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
                print("tote_feature.png 저장")

    finally:
        pipeline.stop()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()