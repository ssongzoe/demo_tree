#!/usr/bin/env python3
"""AR 마커 기준 모바일 베이스 정렬 skill.

동작
1. 마커 측정 → yaw 정렬
2. 마커 재측정 → 전후 / 좌우 위치 정렬

실제 로봇 이동은 control.mobile_controller의 build_leg() / move_leg()을 사용한다.
"""

import math
import time

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
TARGET_MARKER_POS = (0.0, 0.0, 0.30)

MIN_TURN_DEG = 2.0
POSITION_TOL = 0.01
VERTICAL_WARN_M = 0.05

SETTLE_S = 0.7

ALIGN_LINEAR_SPEED = 0.15
ALIGN_ANGULAR_SPEED = 0.5
QUINTIC_PEAK = 1.875
MIN_LEG_TIME = 1.5


def _turn_duration(angle_rad):
    """정렬 회전 시간을 계산한다."""
    return max(QUINTIC_PEAK * abs(angle_rad) / ALIGN_ANGULAR_SPEED, MIN_LEG_TIME)


def _translation_duration(distance):
    """정렬 병진 시간을 계산한다."""
    return max(QUINTIC_PEAK * distance / ALIGN_LINEAR_SPEED, MIN_LEG_TIME)


def _move_relative(robot, monitor, x, y, theta, duration):
    """현재 자세 기준 상대 이동을 한 번 수행한다."""
    leg = build_leg(
        start=odom_pose(monitor.odom),
        target=(x, y, theta),
        absolute=False,
        duration=duration,
        turn_direction="shortest",
    )

    return move_leg(robot, monitor, leg, settle=0.7)


def align_to_marker(
    robot,
    monitor,
    marker_id,
    camera_serial=None,
    target_marker_pos=TARGET_MARKER_POS,
):
    """AR 마커의 yaw와 평면 위치를 순서대로 보정한다."""
    detector = create_detector(ARUCO_DICT)
    pnp_size = MARKER_SIZE * MARKER_SCALE

    camera = RealSenseCamera(CAM_WIDTH, CAM_HEIGHT, CAM_FPS, serial=camera_serial)

    try:
        print("AR 카메라 시작")
        camera.start()

        # 1단계: yaw 정렬
        position, yaw_deg = measure_marker(camera, detector, marker_id, pnp_size)

        if yaw_deg is None:
            print(f"마커 id={marker_id} 검출 실패")
            return False

        print(
            f"AR 측정: x={position[0]:+.3f}, y={position[1]:+.3f}, z={position[2]:+.3f} m, "
            f"yaw={yaw_deg:+.2f} deg"
        )

        if abs(yaw_deg) >= MIN_TURN_DEG:
            angle_rad = math.radians(yaw_deg)

            print(f"Yaw 정렬: {yaw_deg:+.2f} deg")

            if not _move_relative(
                robot,
                monitor,
                x=0.0,
                y=0.0,
                theta=angle_rad,
                duration=_turn_duration(angle_rad),
            ):
                print("Yaw 정렬 실패")
                return False

            time.sleep(SETTLE_S)
        else:
            print("Yaw 오차가 작아 회전을 생략합니다.")

        # 2단계: 위치 정렬
        position, yaw_deg = measure_marker(camera, detector, marker_id, pnp_size)

        if position is None:
            print("Yaw 정렬 후 마커를 다시 찾지 못했습니다.")
            return False

        error = np.asarray(position, dtype=float) - np.asarray(target_marker_pos, dtype=float)

        # 카메라 z 오차 → 로봇 전후, 카메라 x 오차 → 로봇 좌우
        forward = float(error[2])
        lateral = float(-error[0])
        distance = float(np.hypot(forward, lateral))

        print(
            f"위치 오차: forward={forward:+.3f} m, lateral={lateral:+.3f} m, "
            f"vertical={error[1]:+.3f} m, yaw={yaw_deg:+.2f} deg"
        )

        if abs(error[1]) > VERTICAL_WARN_M:
            print(f"주의: vertical 오차 {error[1]:+.3f} m는 모바일 베이스로 보정하지 않습니다.")

        if distance <= POSITION_TOL:
            print("마커 위치가 이미 허용 오차 안에 있습니다.")
            return True

        print(f"위치 정렬: forward={forward:+.3f} m, lateral={lateral:+.3f} m")

        if not _move_relative(
            robot,
            monitor,
            x=forward,
            y=lateral,
            theta=0.0,
            duration=_translation_duration(distance),
        ):
            print("위치 정렬 실패")
            return False

        time.sleep(SETTLE_S)

        # # 최종 위치를 한 번 더 확인한다.
        # final_position, final_yaw_deg = measure_marker(camera, detector, marker_id, pnp_size)

        # if final_position is not None:
        #     final_error = np.asarray(final_position, dtype=float) - np.asarray(target_marker_pos, dtype=float)

        #     print(
        #         f"정렬 완료: forward={final_error[2]:+.3f} m, lateral={-final_error[0]:+.3f} m, "
        #         f"yaw={final_yaw_deg:+.2f} deg"
        #     )
        # else:
        #     print("정렬 이동은 완료했지만 최종 마커 측정은 실패했습니다.")

        return True

    finally:
        camera.stop()
