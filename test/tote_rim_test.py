#!/usr/bin/env python3
"""D435 RGB에서 토트박스 상단 rim의 대표 4개 직선을 검출하는 테스트.

검출 대상
- BACK  : 토트 뒤쪽 rim
- FRONT : 토트 앞쪽 rim
- LEFT  : 토트 왼쪽 rim
- RIGHT : 토트 오른쪽 rim

Depth와 로봇 제어는 사용하지 않는다.
현재 단계의 목적은 RGB 영상에서 대표 네 선이 안정적으로 잡히는지 확인하는 것이다.

사용법
python test/tote_rim_test.py
python test/tote_rim_test.py --serial 250122079439
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
import pyrealsense2 as rs


# ============================================================
# 카메라 설정
# ============================================================

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30


# ============================================================
# Edge / Hough 설정
# ============================================================

CANNY_LOW = 40
CANNY_HIGH = 120

HOUGH_THRESHOLD = 50
MIN_LINE_LENGTH = 70
MAX_LINE_GAP = 35


# ============================================================
# 관심 영역
#
# 현재 카메라에서는 토트가 화면 아래쪽 대부분을 차지하므로
# 위쪽 배경 물체를 최대한 제외한다.
# ============================================================

ROI_LEFT_RATIO = 0.05
ROI_RIGHT_RATIO = 0.95
ROI_TOP_RATIO = 0.40
ROI_BOTTOM_RATIO = 0.98


# ============================================================
# 선 분류 각도
#
# 영상 좌표계 기준 line angle은 0~180 deg.
#
# 현재 영상 예:
# BACK / FRONT : 약 178~179 deg
# LEFT         : 약 106~107 deg
# RIGHT        : 약 65 deg
# ============================================================

HORIZONTAL_MAX_DEG = 15.0

LEFT_MIN_DEG = 90.0
LEFT_MAX_DEG = 135.0

RIGHT_MIN_DEG = 45.0
RIGHT_MAX_DEG = 90.0


# 대표선 fitting에 너무 많은 선을 넣지 않는다.
MAX_GROUP_LINES = 8


def line_info(line):
    """선분의 길이, 각도, 중심점을 계산한다."""
    x1, y1, x2, y2 = [float(value) for value in line]

    dx = x2 - x1
    dy = y2 - y1

    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx)) % 180.0

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    return length, angle, cx, cy


def is_horizontal(angle_deg):
    """0도 또는 180도 근처의 수평선을 판별한다."""
    return (
        angle_deg <= HORIZONTAL_MAX_DEG
        or angle_deg >= 180.0 - HORIZONTAL_MAX_DEG
    )


def fit_representative_line(lines):
    """여러 Hough 선분을 하나의 대표 직선으로 fitting한다."""
    if not lines:
        return None

    # 긴 선을 우선 사용한다.
    selected = sorted(
        lines,
        key=lambda item: item[0],
        reverse=True,
    )[:MAX_GROUP_LINES]

    points = []

    for _, line in selected:
        x1, y1, x2, y2 = line

        points.append([x1, y1])
        points.append([x2, y2])

    points = np.asarray(points, dtype=np.float32)

    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).reshape(-1)

    return np.asarray(
        [vx, vy, x0, y0],
        dtype=np.float64,
    )


def fitted_line_angle_deg(fitted):
    """cv2.fitLine 결과의 line angle을 계산한다."""
    vx, vy, _, _ = fitted
    return math.degrees(math.atan2(vy, vx)) % 180.0


def fitted_line_midpoint_y(fitted, width):
    """화면 중앙 x에서 representative line의 y 위치를 계산한다."""
    vx, vy, x0, y0 = fitted

    x = width * 0.5

    if abs(vx) < 1e-8:
        return float(y0)

    return float(y0 + (x - x0) * vy / vx)


def clip_fitted_line(fitted, roi):
    """무한 직선을 ROI 안에서 그릴 수 있는 두 점으로 변환한다."""
    vx, vy, x0, y0 = fitted

    scale = 2000.0

    p1 = (
        int(round(x0 - scale * vx)),
        int(round(y0 - scale * vy)),
    )

    p2 = (
        int(round(x0 + scale * vx)),
        int(round(y0 + scale * vy)),
    )

    x, y, width, height = roi

    success, clipped_p1, clipped_p2 = cv2.clipLine(
        (x, y, width, height),
        p1,
        p2,
    )

    if not success:
        return None

    return clipped_p1, clipped_p2


def split_horizontal_groups(horizontal_lines, width):
    """수평선 후보를 위쪽 BACK과 아래쪽 FRONT 두 그룹으로 나눈다."""
    if len(horizontal_lines) < 2:
        return [], []

    # 각 선분의 화면 y 중심값.
    values = np.asarray(
        [item[2] for item in horizontal_lines],
        dtype=np.float64,
    )

    # 간단한 1D two-cluster.
    center_a = float(np.min(values))
    center_b = float(np.max(values))

    labels = np.zeros(len(values), dtype=np.int32)

    for _ in range(10):
        distance_a = np.abs(values - center_a)
        distance_b = np.abs(values - center_b)

        labels = (distance_b < distance_a).astype(np.int32)

        group_a = values[labels == 0]
        group_b = values[labels == 1]

        if len(group_a):
            center_a = float(np.mean(group_a))

        if len(group_b):
            center_b = float(np.mean(group_b))

    first = [
        horizontal_lines[index]
        for index in range(len(horizontal_lines))
        if labels[index] == 0
    ]

    second = [
        horizontal_lines[index]
        for index in range(len(horizontal_lines))
        if labels[index] == 1
    ]

    # y가 작은 쪽이 토트 뒤쪽 rim.
    if center_a <= center_b:
        return first, second

    return second, first


def detect_rim_lines(image):
    """RGB 영상에서 BACK / FRONT / LEFT / RIGHT 대표선을 찾는다."""
    height, width = image.shape[:2]

    roi_x0 = int(width * ROI_LEFT_RATIO)
    roi_x1 = int(width * ROI_RIGHT_RATIO)
    roi_y0 = int(height * ROI_TOP_RATIO)
    roi_y1 = int(height * ROI_BOTTOM_RATIO)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(
        gray,
        CANNY_LOW,
        CANNY_HIGH,
    )

    # 끊긴 rim edge를 조금 연결한다.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3),
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    # ROI 밖의 edge는 Hough 입력에서 제거한다.
    roi_mask = np.zeros_like(edges)

    roi_mask[
        roi_y0:roi_y1,
        roi_x0:roi_x1,
    ] = 255

    roi_edges = cv2.bitwise_and(
        edges,
        roi_mask,
    )

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

    lines = np.asarray(
        detected,
        dtype=np.int32,
    ).reshape(-1, 4)

    horizontal = []
    left = []
    right = []

    for line in lines:
        length, angle, cx, cy = line_info(line)

        if length < MIN_LINE_LENGTH:
            continue

        # ----------------------------------------------------
        # BACK / FRONT 후보
        # ----------------------------------------------------

        if is_horizontal(angle):
            # 토트 중앙 부근을 지나는 긴 수평선 위주로 사용한다.
            if (
                width * 0.10 <= cx <= width * 0.90
                and cy >= roi_y0
            ):
                horizontal.append(
                    (length, line.copy(), cy)
                )

            continue

        # ----------------------------------------------------
        # LEFT 후보
        # ----------------------------------------------------

        if (
            LEFT_MIN_DEG <= angle <= LEFT_MAX_DEG
            and cx < width * 0.48
        ):
            left.append(
                (length, line.copy())
            )

            continue

        # ----------------------------------------------------
        # RIGHT 후보
        # ----------------------------------------------------

        if (
            RIGHT_MIN_DEG <= angle < RIGHT_MAX_DEG
            and cx > width * 0.52
        ):
            right.append(
                (length, line.copy())
            )

    back_group, front_group = split_horizontal_groups(
        horizontal,
        width,
    )

    # horizontal group 형식을 fit 함수 형태로 맞춘다.
    back_fit_input = [
        (length, line)
        for length, line, _ in back_group
    ]

    front_fit_input = [
        (length, line)
        for length, line, _ in front_group
    ]

    back_line = fit_representative_line(
        back_fit_input,
    )

    front_line = fit_representative_line(
        front_fit_input,
    )

    left_line = fit_representative_line(
        left,
    )

    right_line = fit_representative_line(
        right,
    )

    result = {
        "back": back_line,
        "front": front_line,
        "left": left_line,
        "right": right_line,
        "roi": (
            roi_x0,
            roi_y0,
            roi_x1 - roi_x0,
            roi_y1 - roi_y0,
        ),
    }

    return result, roi_edges


def draw_result(image, result):
    """최종 대표 4개 직선만 영상에 표시한다."""
    output = image.copy()

    if result is None:
        cv2.putText(
            output,
            "NO RIM LINES",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return output

    roi = result["roi"]

    # ROI는 얇은 회색 사각형으로만 표시한다.
    x, y, width, height = roi

    cv2.rectangle(
        output,
        (x, y),
        (x + width, y + height),
        (150, 150, 150),
        1,
    )

    line_styles = {
        "back": ((0, 255, 255), "BACK"),
        "front": ((255, 0, 255), "FRONT"),
        "left": ((0, 255, 0), "LEFT"),
        "right": ((255, 255, 0), "RIGHT"),
    }

    found_count = 0

    for name in (
        "back",
        "front",
        "left",
        "right",
    ):
        fitted = result[name]

        if fitted is None:
            continue

        clipped = clip_fitted_line(
            fitted,
            roi,
        )

        if clipped is None:
            continue

        color, label = line_styles[name]
        p1, p2 = clipped

        cv2.line(
            output,
            p1,
            p2,
            color,
            4,
            cv2.LINE_AA,
        )

        angle = fitted_line_angle_deg(
            fitted,
        )

        text_x = int(
            0.5 * (p1[0] + p2[0])
        )

        text_y = int(
            0.5 * (p1[1] + p2[1])
        )

        cv2.putText(
            output,
            f"{label} {angle:.1f}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        found_count += 1

    status_color = (
        (0, 255, 0)
        if found_count == 4
        else (0, 0, 255)
    )

    cv2.putText(
        output,
        f"RIM LINES: {found_count}/4",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        2,
        cv2.LINE_AA,
    )

    return output


def start_camera(serial):
    """D435 RGB 스트림만 시작한다."""
    pipeline = rs.pipeline()
    config = rs.config()

    if serial:
        config.enable_device(serial)

    config.enable_stream(
        rs.stream.color,
        CAM_WIDTH,
        CAM_HEIGHT,
        rs.format.bgr8,
        CAM_FPS,
    )

    profile = pipeline.start(
        config,
    )

    # 자동 노출 등이 안정될 시간을 준다.
    for _ in range(30):
        pipeline.wait_for_frames()

    device = profile.get_device()

    try:
        camera_serial = device.get_info(
            rs.camera_info.serial_number
        )
    except Exception:
        camera_serial = "unknown"

    print(
        f"D435 RGB 시작: serial={camera_serial}"
    )

    print(
        f"resolution={CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS}"
    )

    return pipeline


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--serial",
        default=None,
        help="사용할 D435 serial",
    )

    args = parser.parse_args()

    pipeline = start_camera(
        args.serial,
    )

    print("q 또는 ESC: 종료")
    print("s: 현재 결과 저장")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            image = np.asarray(
                color_frame.get_data()
            )

            result, _ = detect_rim_lines(
                image,
            )

            visualized = draw_result(
                image,
                result,
            )

            cv2.imshow(
                "Tote 4 Rim Lines",
                visualized,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                27,
                ord("q"),
                ord("Q"),
            ):
                break

            if key in (
                ord("s"),
                ord("S"),
            ):
                cv2.imwrite(
                    "tote_4_rim_lines.png",
                    visualized,
                )

                cv2.imwrite(
                    "tote_rgb.png",
                    image,
                )

                print(
                    "tote_4_rim_lines.png / tote_rgb.png 저장"
                )

    finally:
        pipeline.stop()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()