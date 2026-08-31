#!/usr/bin/env python3
"""AR 마커 기준 모바일 베이스 one-shot 정렬 skill.

마커 카메라는 주행 중 미리 켜둘 수 있으며, main에서 주입한 공용 카메라를 사용할 수도 있다.
정렬 시점에는 fresh frame만 짧게 모아 x / y / yaw를 한 번에 보정한다.
기존 yaw 이동 → 재측정 → 위치 이동 구조와 달리 실제 로봇 이동은 한 번만 수행한다.
"""

import math
import time

import numpy as np

from control.mobile_controller import build_leg, move_leg, odom_pose
from utils.ar_marker import RealSenseCamera, create_detector, measure_marker


ARUCO_DICT = "DICT_APRILTAG_36h11"

MARKER_SIZE = 0.08
MARKER_SCALE = 0.8

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# 실제 배치 목표 위치에서 측정한 marker pose
TARGET_MARKER_POS = np.array([0.025, 0.069, 0.255], dtype=np.float64)
TARGET_MARKER_YAW_DEG = 1.74

POSITION_TOL = 0.01
YAW_TOL_DEG = 2.0
VERTICAL_WARN_M = 0.05

ALIGN_LINEAR_SPEED = 0.15
ALIGN_ANGULAR_SPEED = 0.5
QUINTIC_PEAK = 1.875
MIN_LEG_TIME = 1.5
SETTLE_S = 0.2

# Tote calibration과 같은 640x480@30 공용 stream에서, 이동 후 쌓인 frame은 조금만 버리고 fresh detection 4개를 측정한다.
MEASURE_FRAMES = 4
FLUSH_FRAMES = 2
MEASURE_TIMEOUT_S = 2.0

# 잘못된 검출값 하나로 비정상적으로 큰 이동을 만드는 경우만 막는 느슨한 guard
MAX_TRANSLATION_M = 0.80
MAX_YAW_DEG = 30.0


def _move_duration(distance: float, angle_rad: float) -> float:
    """병진과 회전을 동시에 수행하므로 두 동작에 필요한 시간 중 큰 값을 trajectory duration으로 사용한다."""
    linear_time = QUINTIC_PEAK * distance / ALIGN_LINEAR_SPEED
    angular_time = QUINTIC_PEAK * abs(angle_rad) / ALIGN_ANGULAR_SPEED
    return max(linear_time, angular_time, MIN_LEG_TIME)


def _one_shot_command(position, yaw_deg: float, target_marker_pos, target_marker_yaw_deg: float):
    """현재 marker pose에서 목표 marker pose가 되도록 로봇 기준 상대 x / y / yaw 명령을 계산한다."""
    current = np.asarray(position, dtype=np.float64)
    target = np.asarray(target_marker_pos, dtype=np.float64)

    # OpenCV camera +z = robot +x(forward), camera +x = robot -y(lateral)
    marker_current_xy = np.array([current[2], -current[0]], dtype=np.float64)
    marker_target_xy = np.array([target[2], -target[0]], dtype=np.float64)

    yaw_error_deg = float(yaw_deg - target_marker_yaw_deg)
    theta = math.radians(yaw_error_deg)

    # x / y / yaw를 동시에 움직여도 최종 marker 위치가 target에 오도록 회전이 위치에 미치는 영향까지 포함한다.
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


class ARAligner:
    """자체 카메라 또는 main에서 주입한 공용 카메라로 fresh marker pose를 측정해 one-shot 정렬한다."""

    def __init__(
        self,
        marker_id,
        camera_serial=None,
        *,
        camera=None,
        target_marker_pos=TARGET_MARKER_POS,
        target_marker_yaw_deg=TARGET_MARKER_YAW_DEG,
    ):
        if camera is not None and camera_serial is not None:
            raise ValueError("공용 camera 사용 시 camera_serial은 함께 지정할 수 없습니다.")

        self.marker_id = marker_id
        self.target_marker_pos = np.asarray(target_marker_pos, dtype=np.float64)
        self.target_marker_yaw_deg = float(target_marker_yaw_deg)

        self.detector = create_detector(ARUCO_DICT)
        self.pnp_size = MARKER_SIZE * MARKER_SCALE
        self._owns_camera = camera is None
        self.camera = camera if camera is not None else RealSenseCamera(CAM_WIDTH, CAM_HEIGHT, CAM_FPS, serial=camera_serial)
        self.started = False

    def start(self) -> None:
        """단독 사용 시에만 AR 카메라를 시작한다. 공용 camera의 start는 main이 담당한다."""
        if self.started:
            return

        if self._owns_camera:
            start_time = time.perf_counter()
            print("AR 카메라 미리 시작")
            self.camera.start()
            print(f"AR 카메라 시작 완료: {time.perf_counter() - start_time:.3f} s")

        self.started = True

    def stop(self) -> None:
        """단독 사용 시에만 AR 카메라 stream을 종료한다."""
        if not self.started:
            return

        if self._owns_camera:
            self.camera.stop()

        self.started = False

    def _measure(self):
        """이동 중 쌓인 frame을 조금 버리고 4개의 fresh detection median을 반환하며 실제 측정 시간을 함께 출력한다."""
        start_time = time.perf_counter()

        position, yaw_deg = measure_marker(
            self.camera,
            self.detector,
            self.marker_id,
            self.pnp_size,
            measure_frames=MEASURE_FRAMES,
            flush_frames=FLUSH_FRAMES,
            timeout_s=MEASURE_TIMEOUT_S,
        )

        print(f"AR marker 측정 시간: {time.perf_counter() - start_time:.3f} s")
        return position, yaw_deg

    def align(self, robot, monitor, verify=False) -> bool:
        """현재 fresh marker pose에서 x / y / yaw를 동시에 계산해 한 번의 SE(2) trajectory로 목표 위치에 정렬한다."""
        if not self.started:
            self.start()

        position, yaw_deg = self._measure()

        if position is None or yaw_deg is None:
            print(f"마커 id={self.marker_id} 검출 실패")
            return False

        target = self.target_marker_pos
        error = np.asarray(position, dtype=np.float64) - target
        x, y, theta, yaw_error_deg = _one_shot_command(position, yaw_deg, target, self.target_marker_yaw_deg)

        distance = float(np.hypot(x, y))
        vertical_error = float(error[1])

        print(f"AR 측정: x={position[0]:+.3f}, y={position[1]:+.3f}, z={position[2]:+.3f} m, yaw={yaw_deg:+.2f} deg")
        print(f"AR 목표: x={target[0]:+.3f}, y={target[1]:+.3f}, z={target[2]:+.3f} m, yaw={self.target_marker_yaw_deg:+.2f} deg")
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

        final_position, final_yaw_deg = self._measure()

        if final_position is None or final_yaw_deg is None:
            print("이동은 완료했지만 최종 marker 확인에 실패했습니다.")
            return True

        final_error = np.asarray(final_position, dtype=np.float64) - target
        final_yaw_error = float(final_yaw_deg - self.target_marker_yaw_deg)

        print(f"최종 확인: x_err={final_error[0]:+.3f} m, z_err={final_error[2]:+.3f} m, yaw_err={final_yaw_error:+.2f} deg")
        return True


def align_to_marker(
    robot,
    monitor,
    marker_id,
    camera_serial=None,
    target_marker_pos=TARGET_MARKER_POS,
    target_marker_yaw_deg=TARGET_MARKER_YAW_DEG,
    verify=False,
):
    """기존 demo_align.py 호환용 함수이며, 단독 호출 시에는 내부에서 카메라 start / align / stop을 모두 수행한다."""
    aligner = ARAligner(
        marker_id=marker_id,
        camera_serial=camera_serial,
        target_marker_pos=target_marker_pos,
        target_marker_yaw_deg=target_marker_yaw_deg,
    )

    try:
        aligner.start()
        return aligner.align(robot, monitor, verify=verify)
    finally:
        aligner.stop()