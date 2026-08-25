#!/usr/bin/env python3
"""D435 TOP rim을 이용해 RB-Y1 base yaw를 한 번에 정렬하는 테스트.

동작
1. TOP rim을 여러 frame 측정한다.
2. median angle을 계산한다.
3. 계산된 yaw 보정량만큼 base를 한 번에 회전한다.
4. 회전 후 TOP rim을 다시 측정해 결과만 검증한다.
5. 검증 결과로 추가 이동하지 않는다.

사용법:
python test/tote_yaw_align_test.py --serial 250122079439
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

ALIGN_ANGULAR_SPEED = 0.5
QUINTIC_PEAK = 1.875
MIN_LEG_TIME = 1.5


# ============================================================
# Yaw 정렬
# ============================================================

# 정렬이 완료된 기준 자세에서 TOP rim이 영상상 거의 0도이므로 0도를 목표로 사용한다.
TARGET_ANGLE_DEG = 0.0

# 처음부터 목표 범위 안이면 base를 움직이지 않는다.
ANGLE_TOL_DEG = 0.8

# 실제 테스트에서 image angle 약 5.083도 보정에 base yaw 약 7.499도가 필요했다.
YAW_GAIN = 1.475

# image angle이 음수일 때 base는 양의 yaw 방향으로 돌아야 한다.
YAW_SIGN = -1.0

# odom 복귀 후 vision이 처리할 수 있는 최대 yaw 오차 범위를 제한한다.
MAX_YAW_COMMAND_DEG = 10.0

# 이동 후 검증에서 이 범위 안이면 SUCCESS로 판단한다.
VERIFY_ANGLE_TOL_DEG = 0.8


# ============================================================
# 카메라
# ============================================================

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

CAMERA_WARMUP_FRAMES = 30
CAMERA_FLUSH_FRAMES = 10


# ============================================================
# TOP rim 검출
# ============================================================

CANNY_LOW = 40
CANNY_HIGH = 120

HOUGH_THRESHOLD = 45
MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 40

MAX_TOP_ANGLE_DEG = 12.0

# odom 복귀 후 테이블 앞 근처에 있다는 전제로 TOP rim이 나타나는 세로 영역을 제한한다.
TOP_MIN_Y = 55
TOP_MAX_Y = 175

# TOP 바로 아래는 검은 토트 내부이므로 위/아래 밝기 차이를 후보 선택에 사용한다.
CONTRAST_OFFSET_PX = 10
CONTRAST_SAMPLE_COUNT = 20

# 한 번의 측정에서 사용할 frame 수.
MEASURE_FRAMES = 20
MEASURE_TIMEOUT_S = 4.0

# 여러 frame에서 측정값이 크게 흔들리면 잘못된 line으로 판단하고 base를 움직이지 않는다.
MAX_ANGLE_STD_DEG = 1.0
MAX_CENTER_Y_STD_PX = 4.0


@dataclass
class TopFeature:
    """한 frame에서 검출한 TOP rim."""

    angle_deg: float
    center_x_px: float
    center_y_px: float
    length_px: float
    contrast: float
    p1: tuple[int, int]
    p2: tuple[int, int]


@dataclass
class TopMeasurement:
    """여러 frame을 모아서 계산한 안정화된 TOP 측정값."""

    angle_deg: float
    center_x_px: float
    center_y_px: float
    angle_std_deg: float
    center_y_std_px: float
    valid_frames: int


def normalize_horizontal_angle_deg(angle_deg: float) -> float:
    """직선 각도를 수평 기준 -90~90도 범위로 변환한다."""
    angle = (float(angle_deg) + 180.0) % 180.0

    if angle >= 90.0:
        angle -= 180.0

    return angle


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


def line_y_at_x(line: np.ndarray, x: float) -> float | None:
    """선분을 무한 직선으로 보고 지정한 x 위치의 y를 계산한다."""
    x1, y1, x2, y2 = [float(value) for value in line]

    dx = x2 - x1

    if abs(dx) < 1e-8:
        return None

    return y1 + (float(x) - x1) * (y2 - y1) / dx


def sample_line_contrast(gray: np.ndarray, line: np.ndarray) -> float:
    """TOP 바로 위와 아래 밝기 차이를 계산한다. 양수일수록 아래쪽이 더 어둡다."""
    height, width = gray.shape

    x1, _, x2, _ = [float(value) for value in line]

    start_x = min(x1, x2)
    end_x = max(x1, x2)

    xs = np.linspace(start_x, end_x, CONTRAST_SAMPLE_COUNT)

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


def detect_top_rim(image: np.ndarray) -> tuple[TopFeature | None, np.ndarray]:
    """수평선 후보 중 길이와 TOP 아래쪽 밝기 조건을 이용해 가장 좋은 rim 하나를 선택한다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    roi_edges = np.zeros_like(edges)
    roi_edges[TOP_MIN_Y:TOP_MAX_Y, :] = edges[TOP_MIN_Y:TOP_MAX_Y, :]

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

    best_feature = None
    best_score = -float("inf")

    for line in lines:
        length, angle_deg, center_x, center_y = line_info(line)

        if length < MIN_LINE_LENGTH or abs(angle_deg) > MAX_TOP_ANGLE_DEG:
            continue

        if not TOP_MIN_Y <= center_y <= TOP_MAX_Y:
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

        best_feature = TopFeature(
            angle_deg=angle_deg,
            center_x_px=center_x,
            center_y_px=center_y,
            length_px=length,
            contrast=contrast,
            p1=p1,
            p2=p2,
        )

    return best_feature, roi_edges


def draw_feature(image: np.ndarray, feature: TopFeature | None, title: str) -> np.ndarray:
    """현재 TOP 검출 결과를 화면에 표시한다."""
    output = image.copy()

    cv2.line(output, (0, TOP_MIN_Y), (CAM_WIDTH - 1, TOP_MIN_Y), (100, 100, 100), 1)
    cv2.line(output, (0, TOP_MAX_Y), (CAM_WIDTH - 1, TOP_MAX_Y), (100, 100, 100), 1)

    if feature is None:
        cv2.putText(output, "TOP NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return output

    center = (int(round(feature.center_x_px)), int(round(feature.center_y_px)))

    cv2.line(output, feature.p1, feature.p2, (0, 255, 255), 4, cv2.LINE_AA)
    cv2.circle(output, center, 7, (0, 0, 255), -1)

    text = (
        f"{title}  angle={feature.angle_deg:+.2f} deg  cx={feature.center_x_px:.1f}  "
        f"cy={feature.center_y_px:.1f}  contrast={feature.contrast:+.1f}"
    )

    cv2.putText(output, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA)

    return output


def measure_top(pipeline: rs.pipeline, title: str) -> TopMeasurement | None:
    """여러 frame의 TOP 검출 결과를 모아서 median 측정값을 계산한다."""
    features = []
    start = time.monotonic()

    while len(features) < MEASURE_FRAMES and time.monotonic() - start < MEASURE_TIMEOUT_S:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asarray(color_frame.get_data())
        feature, _ = detect_top_rim(image)

        cv2.imshow("Tote Yaw Align", draw_feature(image, feature, title))
        cv2.waitKey(1)

        if feature is not None:
            features.append(feature)

    if len(features) < MEASURE_FRAMES // 2:
        print(f"TOP 측정 실패: valid={len(features)}/{MEASURE_FRAMES}")
        return None

    center_ys = np.asarray([feature.center_y_px for feature in features], dtype=np.float64)
    median_y = float(np.median(center_ys))

    # 순간적으로 다른 수평선을 잡은 frame은 center_y를 기준으로 제거한다.
    filtered = [feature for feature in features if abs(feature.center_y_px - median_y) <= 5.0]

    if len(filtered) < MEASURE_FRAMES // 2:
        print(f"TOP 안정화 실패: filtered={len(filtered)}/{len(features)}")
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

    if measurement.angle_std_deg > MAX_ANGLE_STD_DEG:
        print("angle 측정이 불안정해서 이동하지 않습니다.")
        return None

    if measurement.center_y_std_px > MAX_CENTER_Y_STD_PX:
        print("TOP line 위치가 불안정해서 이동하지 않습니다.")
        return None

    return measurement


def flush_camera(pipeline: rs.pipeline, frames: int = CAMERA_FLUSH_FRAMES) -> None:
    """로봇 이동 중 카메라에 쌓인 이전 frame을 버린다."""
    for _ in range(frames):
        pipeline.wait_for_frames()


def turn_duration(angle_rad: float) -> float:
    """작은 yaw 이동에 필요한 trajectory 시간을 계산한다."""
    return max(QUINTIC_PEAK * abs(angle_rad) / ALIGN_ANGULAR_SPEED, MIN_LEG_TIME)


def move_relative_yaw(robot, monitor: OdometryMonitor, angle_deg: float) -> bool:
    """현재 odom 자세를 기준으로 yaw만 상대 이동한다."""
    angle_rad = math.radians(angle_deg)

    leg = build_leg(
        start=odom_pose(monitor.odom),
        target=(0.0, 0.0, angle_rad),
        absolute=False,
        duration=turn_duration(angle_rad),
        turn_direction="shortest",
    )

    return move_leg(robot, monitor, leg, settle=SETTLE_S)


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


def clamp(value: float, minimum: float, maximum: float) -> float:
    """값을 지정 범위로 제한한다."""
    return max(minimum, min(maximum, value))


def align_yaw_once(robot, monitor: OdometryMonitor, pipeline: rs.pipeline) -> bool:
    """TOP rim을 한 번 측정하고 계산된 yaw를 한 번에 보정한 뒤 결과만 검증한다."""
    print()
    print("========== YAW ALIGN ==========")

    before = measure_top(pipeline, "BEFORE")

    if before is None:
        return False

    error_deg = before.angle_deg - TARGET_ANGLE_DEG

    print(
        f"Yaw error: current={before.angle_deg:+.3f} deg, "
        f"target={TARGET_ANGLE_DEG:+.3f} deg, error={error_deg:+.3f} deg"
    )

    if abs(error_deg) <= ANGLE_TOL_DEG:
        print("이미 Yaw 정렬 범위 안입니다.")
        return True

    correction_deg = YAW_SIGN * YAW_GAIN * error_deg
    correction_deg = clamp(correction_deg, -MAX_YAW_COMMAND_DEG, MAX_YAW_COMMAND_DEG)

    print(f"Yaw gain       : {YAW_GAIN:.3f}")
    print(f"Base yaw command: {correction_deg:+.3f} deg")

    if not move_relative_yaw(robot, monitor, correction_deg):
        print("Base yaw 이동 실패")
        return False

    time.sleep(SETTLE_S)
    flush_camera(pipeline)

    print()
    print("========== YAW VERIFY ==========")

    after = measure_top(pipeline, "AFTER")

    if after is None:
        print("Yaw 이동은 완료했지만 최종 Vision 검증에 실패했습니다.")
        return False

    final_error_deg = after.angle_deg - TARGET_ANGLE_DEG

    print()
    print(f"Before angle : {before.angle_deg:+.3f} deg")
    print(f"Command      : {correction_deg:+.3f} deg")
    print(f"After angle  : {after.angle_deg:+.3f} deg")
    print(f"Final error  : {final_error_deg:+.3f} deg")

    if abs(final_error_deg) <= VERIFY_ANGLE_TOL_DEG:
        print("Yaw 1회 정렬 성공")
        return True

    print("Yaw 1회 이동은 완료했지만 최종 오차가 tolerance 밖입니다.")
    print("추가 보정은 수행하지 않습니다.")

    return False


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
        print(f"TARGET_ANGLE_DEG     = {TARGET_ANGLE_DEG:+.3f}")
        print(f"ANGLE_TOL_DEG        = ±{ANGLE_TOL_DEG:.3f}")
        print(f"YAW_GAIN             = {YAW_GAIN:.3f}")
        print(f"MAX_YAW_COMMAND_DEG  = ±{MAX_YAW_COMMAND_DEG:.1f}")
        print()
        print("Yaw 1회 정렬 테스트 시작")

        success = align_yaw_once(robot, monitor, pipeline)

        print()
        print("RESULT:", "SUCCESS" if success else "FAILED")

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