#!/usr/bin/env python3
"""D435 RGB 영상에서 열린 토트박스 상단 rim 후보를 검출하는 테스트.

목적
- Depth는 사용하지 않는다.
- RGB edge/contour에서 가장 그럴듯한 사각형을 찾는다.
- 검출된 4개 corner, 중심점, 영상 기준 긴 변 yaw를 화면에 표시한다.
- 이 단계에서는 로봇을 움직이지 않는다.

사용법
python test_tote_rim.py
python test_tote_rim.py --serial 123456789
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
import pyrealsense2 as rs


CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

MIN_CONTOUR_AREA = 8_000
MAX_CONTOUR_AREA_RATIO = 0.85

CANNY_LOW = 50
CANNY_HIGH = 150


def order_corners(points: np.ndarray) -> np.ndarray:
    """사각형 네 점을 TL, TR, BR, BL 순서로 정렬한다."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)

    result = np.zeros((4, 2), dtype=np.float32)

    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)

    result[0] = points[np.argmin(point_sum)]    # top-left
    result[2] = points[np.argmax(point_sum)]    # bottom-right
    result[1] = points[np.argmin(point_diff)]   # top-right
    result[3] = points[np.argmax(point_diff)]   # bottom-left

    return result


def polygon_angle_score(points: np.ndarray) -> float:
    """네 꼭짓점이 직사각형에 얼마나 가까운지 0~1 점수로 반환한다."""
    points = order_corners(points)

    cosines = []

    for index in range(4):
        current = points[index]
        previous = points[(index - 1) % 4]
        following = points[(index + 1) % 4]

        v1 = previous - current
        v2 = following - current

        denominator = np.linalg.norm(v1) * np.linalg.norm(v2)

        if denominator < 1e-6:
            return 0.0

        cosine = abs(float(np.dot(v1, v2) / denominator))
        cosines.append(cosine)

    # 90도면 cos=0이므로 점수 1.0에 가까워진다.
    return max(0.0, 1.0 - float(np.mean(cosines)))


def quadrilateral_yaw_deg(points: np.ndarray) -> float:
    """영상에서 긴 변의 각도를 0~180도 line angle로 계산한다."""
    points = order_corners(points)

    edges = [
        points[1] - points[0],
        points[2] - points[1],
        points[3] - points[2],
        points[0] - points[3],
    ]

    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    longest = edges[int(np.argmax(lengths))]

    yaw = math.degrees(math.atan2(float(longest[1]), float(longest[0])))

    return yaw % 180.0


def find_tote_rim(image: np.ndarray):
    """RGB 영상에서 가장 큰 사각형 rim 후보 하나를 찾는다."""
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 끊어진 rim edge를 연결한다.
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_area = float(width * height)

    best_corners = None
    best_score = -1.0

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < MIN_CONTOUR_AREA:
            continue

        if area > image_area * MAX_CONTOUR_AREA_RATIO:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0.0:
            continue

        # contour를 4각형으로 근사한다.
        approximate = cv2.approxPolyDP(
            contour,
            0.025 * perimeter,
            True,
        )

        if len(approximate) != 4:
            continue

        if not cv2.isContourConvex(approximate):
            continue

        corners = approximate.reshape(4, 2).astype(np.float32)

        angle_score = polygon_angle_score(corners)

        # 너무 찌그러진 사각형은 제외한다.
        if angle_score < 0.55:
            continue

        # 큰 사각형을 우선하되 직사각형 형태도 같이 본다.
        score = area * (0.5 + 0.5 * angle_score)

        if score > best_score:
            best_score = score
            best_corners = order_corners(corners)

    return best_corners, edges


def draw_result(
    image: np.ndarray,
    corners: np.ndarray | None,
) -> np.ndarray:
    """검출 결과를 원본 영상 위에 표시한다."""
    output = image.copy()

    if corners is None:
        cv2.putText(
            output,
            "TOTE NOT FOUND",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    polygon = np.rint(corners).astype(np.int32)

    cv2.polylines(
        output,
        [polygon.reshape(-1, 1, 2)],
        True,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )

    names = ("TL", "TR", "BR", "BL")

    for name, point in zip(names, polygon):
        x, y = int(point[0]), int(point[1])

        cv2.circle(
            output,
            (x, y),
            6,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            output,
            name,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    center = np.mean(corners, axis=0)

    cx = int(round(float(center[0])))
    cy = int(round(float(center[1])))

    cv2.circle(
        output,
        (cx, cy),
        8,
        (255, 0, 255),
        -1,
    )

    yaw_deg = quadrilateral_yaw_deg(corners)

    cv2.putText(
        output,
        f"center=({cx}, {cy})  image yaw={yaw_deg:.1f} deg",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return output


def start_camera(serial: str | None):
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

    profile = pipeline.start(config)

    # 자동 노출 등이 안정될 시간을 준다.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=None, help="사용할 D435 serial")
    args = parser.parse_args()

    pipeline = start_camera(args.serial)

    print("q 또는 ESC: 종료")
    print("s: 현재 프레임 저장")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            image = np.asarray(color_frame.get_data())

            corners, edges = find_tote_rim(image)

            result = draw_result(image, corners)

            cv2.imshow("Tote Rim Detection", result)
            cv2.imshow("Edges", edges)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            if key in (ord("s"), ord("S")):
                cv2.imwrite("tote_rgb.png", image)
                cv2.imwrite("tote_result.png", result)
                cv2.imwrite("tote_edges.png", edges)

                print("tote_rgb.png / tote_result.png / tote_edges.png 저장")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()