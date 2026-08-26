#!/usr/bin/env python3
"""AR 마커 기준 모바일 베이스 one-shot 정렬 skill.

동작
1. 마커를 한 번 측정한다.
2. 현재 marker pose와 목표 marker pose 차이에서 로봇 x / y / yaw 명령을 동시에 계산한다.
3. build_leg()에 x / y / yaw를 한 번에 넣어 하나의 SE(2) trajectory로 이동한다.

기존 yaw 이동 → 재측정 → 위치 이동 구조와 달리 실제 로봇 이동은 한 번만 수행한다.
"""

import math

import numpy as np

from control.mobile_controller import build_leg, move_leg, odom_pose
from utils.ar_marker import RealSenseCamera, create_detector, measure_marker


ARUCO_DICT = "DICT_APRILTAG_36h11"

MARKER_SIZE = 0.08
MARKER_SCALE = 0.8

CAM_WIDTH = 848
CAM_HEIGHT = 480
CAM_FPS = 15

# OpenCV 카메라 좌표계: +x 오른쪽, +y 아래쪽, +z 전방
TARGET_MARKER_POS = np.array([0.025, 0.069, 0.255], dtype=np.float64)
TARGET_MARKER_YAW_DEG = 1.74

POSITION_TOL = 0.01
YAW_TOL_DEG = 2.0
VERTICAL_WARN_M = 0.05

ALIGN_LINEAR_SPEED = 0.15
ALIGN_ANGULAR_SPEED = 0.5
QUINTIC_PEAK = 1.875
MIN_LEG_TIME = 1.5
SETTLE_S = 0.7

MAX_TRANSLATION_M = 0.80
MAX_YAW_DEG = 30.0


def _move_duration(distance: float, angle_rad: float) -> float:
    """병진과 회전을 동시에 수행할 수 있도록 필요한 시간 중 큰 값을 trajectory duration으로 사용한다."""
    linear_time = QUINTIC_PEAK * distance / ALIGN_LINEAR_SPEED
    angular_time = QUINTIC_PEAK * abs(angle_rad) / ALIGN_ANGULAR_SPEED
    return max(linear_time, angular_time, MIN_LEG_TIME)


def _one_shot_command(position, yaw_deg: float, target_marker_pos, target_marker_yaw_deg: float):
    """현재 marker pose에서 목표 marker pose가 되도록 로봇 기준 상대 x / y / yaw 명령을 계산한다."""
    current = np.asarray(position, dtype=np.float64)
    target = np.asarray(target_marker_pos, dtype=np.float64)

    # 기존 ar_align 좌표 대응을 그대로 사용한다: camera +z = robot +x, camera +x = robot -y.
    marker_current_xy = np.array([current[2], -current[0]], dtype=np.float64)
    marker_target_xy = np.array([target[2], -target[0]], dtype=np.float64)

    yaw_error_deg = float(yaw_deg - target_marker_yaw_deg)
    theta = math.radians(yaw_error_deg)

    # 회전과 병진을 한 번에 수행해도 marker가 목표 위치에 오도록 목표 marker 위치를 현재 robot frame으로 회전시킨다.
    c = math.cos(theta)
    s = math.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    command_xy = marker_current_xy - rotation @ marker_target_xy

    return float(command_xy[0]), float(command_xy[1]), theta, yaw_error_deg


def _move_relative(robot, monitor, x: float, y: float, theta: float, duration: float) -> bool:
    """현재 odometry 자세 기준 x / y / yaw를 하나의 상대 SE(2) trajectory로 동시에 이동한다."""
    leg = build_leg(
        start=odom_pose(monitor.odom),
        target=(x, y, theta),
        absolute=False,
        duration=duration,
        turn_direction="shortest",
    )

    return move_leg(robot, monitor, leg, settle=SETTLE_S)


def align_to_marker(
    robot,
    monitor,
    marker_id,
    camera_serial=None,
    target_marker_pos=TARGET_MARKER_POS,
    target_marker_yaw_deg=TARGET_MARKER_YAW_DEG,
    verify=False,
):
    """마커를 한 번 측정한 뒤 x / y / yaw를 동시에 보정하여 one-shot으로 목표 marker pose에 정렬한다."""
    detector = create_detector(ARUCO_DICT)
    pnp_size = MARKER_SIZE * MARKER_SCALE
    camera = RealSenseCamera(CAM_WIDTH, CAM_HEIGHT, CAM_FPS, serial=camera_serial)

    try:
        print("AR 카메라 시작")
        camera.start()

        position, yaw_deg = measure_marker(camera, detector, marker_id, pnp_size)

        if position is None or yaw_deg is None:
            print(f"마커 id={marker_id} 검출 실패")
            return False

        target = np.asarray(target_marker_pos, dtype=np.float64)
        error = np.asarray(position, dtype=np.float64) - target
        x, y, theta, yaw_error_deg = _one_shot_command(position, yaw_deg, target, target_marker_yaw_deg)

        distance = float(np.hypot(x, y))
        vertical_error = float(error[1])

        print(f"AR 측정: x={position[0]:+.3f}, y={position[1]:+.3f}, z={position[2]:+.3f} m, yaw={yaw_deg:+.2f} deg")
        print(f"AR 목표: x={target[0]:+.3f}, y={target[1]:+.3f}, z={target[2]:+.3f} m, yaw={target_marker_yaw_deg:+.2f} deg")
        print(f"One-shot command: x={x:+.3f} m, y={y:+.3f} m, yaw={yaw_error_deg:+.2f} deg")

        if abs(vertical_error) > VERTICAL_WARN_M:
            print(f"주의: vertical 오차 {vertical_error:+.3f} m는 모바일 베이스로 보정하지 않습니다.")

        if distance <= POSITION_TOL and abs(yaw_error_deg) <= YAW_TOL_DEG:
            print("AR marker가 이미 허용 오차 안에 있습니다.")
            return True

        if distance > MAX_TRANSLATION_M or abs(yaw_error_deg) > MAX_YAW_DEG:
            print(f"AR command가 비정상적으로 큽니다: translation={distance:.3f} m, yaw={yaw_error_deg:+.2f} deg")
            return False

        duration = _move_duration(distance, theta)
        print(f"AR one-shot 이동 시작: duration={duration:.2f} s")

        if not _move_relative(robot, monitor, x=x, y=y, theta=theta, duration=duration):
            print("AR one-shot 정렬 실패")
            return False

        if not verify:
            print("AR one-shot 정렬 완료")
            return True

        final_position, final_yaw_deg = measure_marker(camera, detector, marker_id, pnp_size)

        if final_position is None or final_yaw_deg is None:
            print("이동은 완료했지만 최종 marker 확인에 실패했습니다.")
            return True

        final_error = np.asarray(final_position, dtype=np.float64) - target
        final_yaw_error = float(final_yaw_deg - target_marker_yaw_deg)

        print(
            f"최종 확인: x_err={final_error[0]:+.3f} m, z_err={final_error[2]:+.3f} m, "
            f"yaw_err={final_yaw_error:+.2f} deg"
        )

        return True

    finally:
        camera.stop()