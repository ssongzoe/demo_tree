#!/usr/bin/env python3
"""D435 RGB에서 토트박스 rim의 긴 직선을 확인하는 테스트."""

import argparse
import math

import cv2
import numpy as np
import pyrealsense2 as rs


CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

CANNY_LOW = 40
CANNY_HIGH = 120

MIN_LINE_LENGTH = 80
MAX_LINE_GAP = 30
HOUGH_THRESHOLD = 50


def detect_tote_lines(image):
    """RGB 영상에서 토트 rim 후보 직선들을 찾는다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(
        gray,
        CANNY_LOW,
        CANNY_HIGH,
    )

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

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    return edges, lines


def line_info(line):
    """선분의 길이와 영상 기준 각도를 계산한다."""
    x1, y1, x2, y2 = line

    dx = x2 - x1
    dy = y2 - y1

    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx))

    # line 방향은 180도 대칭
    angle = (angle + 180.0) % 180.0

    return length, angle


def draw_lines(image, lines):
    """검출된 긴 직선을 화면에 표시한다."""
    output = image.copy()

    if lines is None:
        cv2.putText(
            output,
            "NO LONG LINES",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return output

    candidates = []

    for detected in lines:
        line = detected[0]

        length, angle = line_info(line)

        candidates.append(
            (length, angle, line)
        )

    # 긴 선부터 표시
    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    # 너무 많이 그리면 보기 힘드니 상위 20개만
    for index, (length, angle, line) in enumerate(candidates[:20]):
        x1, y1, x2, y2 = line

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cv2.putText(
            output,
            f"{angle:.0f}",
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        print(
            f"{index:02d}: "
            f"len={length:6.1f}px  "
            f"angle={angle:6.1f}deg  "
            f"({x1},{y1}) -> ({x2},{y2})"
        )

    return output


def start_camera(serial):
    """D435 RGB 스트림을 시작한다."""
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

    pipeline.start(config)

    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=None)
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

            edges, lines = detect_tote_lines(image)
            result = draw_lines(image, lines)

            cv2.imshow("Tote Lines", result)
            cv2.imshow("Edges", edges)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            if key in (ord("s"), ord("S")):
                cv2.imwrite("tote_lines.png", result)
                cv2.imwrite("tote_edges.png", edges)
                print("결과 저장 완료")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()