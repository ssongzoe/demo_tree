#!/usr/bin/env python3
"""AR 마커 기준 모바일 베이스 제한 반복 정렬 skill.

마커 카메라는 주행 중 미리 켜둘 수 있으며,
main에서 주입한 공용 카메라를 사용할 수도 있다.
정렬 시점에는 fresh frame만 짧게 모아 x / y / yaw를 보정한다.
큰 명령은 안전한 step으로 제한하고, 이동 후 재측정하며 최대 3회 보정한다.
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
TARGET_MARKER_POS = np.array([0.025, 0.069, 0.20], dtype=np.float64)
TARGET_MARKER_YAW_DEG = 1.74

POSITION_TOL = 0.02
YAW_TOL_DEG = 3.0
VERTICAL_WARN_M = 0.05

ALIGN_LINEAR_SPEED = 0.15
ALIGN_ANGULAR_SPEED = 0.5
QUINTIC_PEAK = 1.875
MIN_LEG_TIME = 1.0
SETTLE_S = 0.2

# Tote calibration과 같은 640x480@30 공용 stream에서 fresh detection 4개를 측정한다.
MEASURE_FRAMES = 4
FLUSH_FRAMES = 2
MEASURE_TIMEOUT_S = 4.0
MEASURE_ATTEMPTS = 3

# 한 번에 움직일 soft limit과 오검출을 차단할 hard limit을 분리한다.
MAX_CORRECTIONS = 3
MAX_TRANSLATION_STEP_M = 0.35
MAX_YAW_STEP_DEG = 20.0
HARD_MAX_TRANSLATION_M = 1.00
HARD_MAX_YAW_DEG = 60.0


def _wrap_angle_deg(angle_deg: float) -> float:
    """각도 차이를 -180~180도 범위로 정규화한다."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _move_duration(distance: float, angle_rad: float) -> float:
    """병진과 회전에 필요한 시간 중 큰 값을 trajectory duration으로 사용한다."""
    linear_time = QUINTIC_PEAK * distance / ALIGN_LINEAR_SPEED
    angular_time = QUINTIC_PEAK * abs(angle_rad) / ALIGN_ANGULAR_SPEED
    return max(linear_time, angular_time, MIN_LEG_TIME)


def _one_shot_command(position, yaw_deg: float, target_marker_pos, target_marker_yaw_deg: float):
    """현재 marker pose에서 목표가 되도록 로봇 기준 상대 x/y/yaw 명령을 계산한다."""
    current = np.asarray(position, dtype=np.float64)
    target = np.asarray(target_marker_pos, dtype=np.float64)

    # OpenCV camera +z = robot +x(forward), camera +x = robot -y(lateral)
    marker_current_xy = np.array([current[2], -current[0]], dtype=np.float64)
    marker_target_xy = np.array([target[2], -target[0]], dtype=np.float64)

    yaw_error_deg = _wrap_angle_deg(yaw_deg - target_marker_yaw_deg)
    theta = math.radians(yaw_error_deg)

    # 동시 x/y/yaw 이동의 최종 marker 위치가 target에 오도록
    # 회전이 위치에 미치는 영향까지 포함한다.
    c = math.cos(theta)
    s = math.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    command_xy = marker_current_xy - rotation @ marker_target_xy

    return float(command_xy[0]), float(command_xy[1]), theta, yaw_error_deg


def _limit_command_step(x: float, y: float, yaw_error_deg: float):
    """큰 yaw는 회전만 먼저 수행하고, 그 외에는 병진 명령을 soft limit으로 제한한다."""
    if abs(yaw_error_deg) > MAX_YAW_STEP_DEG:
        limited_yaw_deg = math.copysign(MAX_YAW_STEP_DEG, yaw_error_deg)
        return 0.0, 0.0, math.radians(limited_yaw_deg), limited_yaw_deg, "yaw-only"

    distance = math.hypot(x, y)
    scale = min(1.0, MAX_TRANSLATION_STEP_M / distance) if distance > 1e-8 else 1.0
    limited_x = x * scale
    limited_y = y * scale

    return limited_x, limited_y, math.radians(yaw_error_deg), yaw_error_deg, "combined"


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
    """자체 카메라 또는 main의 공용 카메라로 marker를 측정하고 최대 3회 보정한다."""

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
        self.camera = (
            camera
            if camera is not None
            else RealSenseCamera(CAM_WIDTH, CAM_HEIGHT, CAM_FPS, serial=camera_serial)
        )
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
        """이전 frame을 버리고 4개 fresh detection median과 측정 시간을 반환한다."""
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

    def _measure_with_retry(self):
        """마커 검출 실패 시 베이스를 움직이지 않고 현재 자세에서 다시 측정한다."""
        for attempt in range(1, MEASURE_ATTEMPTS + 1):
            position, yaw_deg = self._measure()

            if position is not None and yaw_deg is not None:
                return position, yaw_deg

            if attempt < MEASURE_ATTEMPTS:
                print(f"마커 id={self.marker_id} 검출 실패: 카메라 재측정 {attempt + 1}/{MEASURE_ATTEMPTS}")

        return None, None

    def align(self, robot, monitor, verify=True) -> bool:
        """fresh marker pose를 반복 측정하며 x/y/yaw를 최대 3회 제한 보정한다."""
        if not self.started:
            self.start()

        target = self.target_marker_pos

        for correction_count in range(MAX_CORRECTIONS + 1):
            position, yaw_deg = self._measure_with_retry()

            if position is None or yaw_deg is None:
                print(
                    f"AR 정렬 실패: 마커 id={self.marker_id}를 "
                    f"{MEASURE_ATTEMPTS}회 측정하지 못했습니다."
                )
                return False

            error = np.asarray(position, dtype=np.float64) - target
            x, y, _, yaw_error_deg = _one_shot_command(
                position,
                yaw_deg,
                target,
                self.target_marker_yaw_deg,
            )
            distance = math.hypot(x, y)
            vertical_error = float(error[1])

            print(
                f"AR 측정: x={position[0]:+.3f}, y={position[1]:+.3f}, "
                f"z={position[2]:+.3f} m, yaw={yaw_deg:+.2f} deg"
            )
            print(
                f"AR 목표: x={target[0]:+.3f}, y={target[1]:+.3f}, "
                f"z={target[2]:+.3f} m, yaw={self.target_marker_yaw_deg:+.2f} deg"
            )
            print(f"Raw command: x={x:+.3f} m, y={y:+.3f} m, yaw={yaw_error_deg:+.2f} deg")

            if abs(vertical_error) > VERTICAL_WARN_M:
                print(
                    f"주의: vertical 오차 {vertical_error:+.3f} m는 "
                    "모바일 베이스로 보정하지 않습니다."
                )

            if distance <= POSITION_TOL and abs(yaw_error_deg) <= YAW_TOL_DEG:
                if correction_count == 0:
                    print("AR marker가 이미 허용 오차 안에 있습니다.")
                else:
                    print(f"AR 정렬 성공: {correction_count}회 보정")
                return True

            if correction_count >= MAX_CORRECTIONS:
                print(f"AR 정렬 실패: {MAX_CORRECTIONS}회 보정 후에도 허용 오차를 벗어났습니다.")
                return False

            if distance > HARD_MAX_TRANSLATION_M or abs(yaw_error_deg) > HARD_MAX_YAW_DEG:
                print(
                    "AR 정렬 실패: hard limit을 넘는 명령입니다. "
                    f"translation={distance:.3f} m, yaw={yaw_error_deg:+.2f} deg"
                )
                return False

            move_x, move_y, move_theta, move_yaw_deg, mode = _limit_command_step(x, y, yaw_error_deg)
            move_distance = math.hypot(move_x, move_y)
            correction_number = correction_count + 1
            duration = _move_duration(move_distance, move_theta)

            if mode == "yaw-only":
                print(
                    f"보정 {correction_number}/{MAX_CORRECTIONS}: 큰 yaw이므로 위치 이동 없이 "
                    f"yaw={move_yaw_deg:+.2f} deg만 먼저 보정합니다."
                )
            else:
                print(
                    f"보정 {correction_number}/{MAX_CORRECTIONS}: "
                    f"x={move_x:+.3f} m, y={move_y:+.3f} m, yaw={move_yaw_deg:+.2f} deg"
                )

            print(f"AR 보정 이동 시작: duration={duration:.2f} s")

            if not _move_relative(
                robot,
                monitor,
                x=move_x,
                y=move_y,
                theta=move_theta,
                duration=duration,
            ):
                print(f"AR 정렬 실패: 모바일 보정 {correction_number}/{MAX_CORRECTIONS} 이동 실패")
                return False

            if not verify:
                print("AR one-shot 정렬 완료: verify=False")
                return True

        return False


def align_to_marker(
    robot,
    monitor,
    marker_id,
    camera_serial=None,
    target_marker_pos=TARGET_MARKER_POS,
    target_marker_yaw_deg=TARGET_MARKER_YAW_DEG,
    verify=True,
):
    """기존 demo 호환용이며 단독 호출 시 카메라 start/align/stop을 수행한다."""
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