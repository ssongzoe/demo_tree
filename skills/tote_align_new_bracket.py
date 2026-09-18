#!/usr/bin/env python3
"""새 헤드 브라켓용 tote 정렬기.

한 측정 window에서 [corner_x, corner_y, top_angle]을 얻고, 새 브라켓에서
측정한 3x3 local calibration 역행렬로 [x, y, yaw] 오차를 동시에 계산한다.
각 보정도 x/y/yaw를 하나의 SE(2) trajectory로 함께 보낸다.

정책:
- Head 자세는 calibration과 같은 [0, 43] deg를 사용한다.
- 첫 신뢰 측정에서 RIGHT를 우선하고, 선택한 side를 정렬 종료까지 고정한다.
- RIGHT로 보정한 뒤 RIGHT가 잠시 안 보이면 LEFT로 바꾸지 않고 같은 자리에서 재측정한다.
- 첫 명령 뒤 재측정하고, 필요할 때만 한 번 더 보정한다.
- 첫 보정 전까지 신뢰할 측정이 없으면 전방 3 cm recovery를 최대 두 번 수행한다.
- 검출 실패 때 로봇을 탐색 이동시키지 않고 같은 자리에서 다시 측정한다.
- command stream이 만료되면 같은 world 목표의 남은 이동만 한 번 재전송한다.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np


# skills/ 또는 프로젝트 root 어디에 놓아도 실행되도록 한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "control").is_dir() else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.mobile_controller import (  # noqa: E402
    OdometryMonitor,
    build_leg,
    initialize_mobile,
    move_leg,
    odom_pose,
    wait_for_odometry,
)
from control.robot_controller import move_both_arms, move_torso_and_head  # noqa: E402
from utils.ar_marker import RealSenseCamera  # noqa: E402
from utils.tote_vision_new_bracket import (  # noqa: E402
    detect_frame_feature,
    draw_feature,
    flush_camera,
    reset_top_tracking,
)


# Head [0, 43] deg로 다시 측정한 최종 파지 기준값.
TARGET_LEFT = np.asarray([80.335, 158.664, -2.8361], dtype=np.float64)
TARGET_RIGHT = np.asarray([579.906, 134.103, -2.8361], dtype=np.float64)

# [feature_x error(px), feature_y error(px), angle error(deg)]
# -> [robot_x error(m), robot_y error(m), robot_yaw error(deg)]
POSE_FROM_FEATURE_LEFT = np.asarray(
    [
        [-0.0002730845, 0.0014462735, 0.0106549943],
        [0.0011204132, 0.0009572346, -0.0113593638],
        [0.0020248340, 0.0037553196, 1.6082597500],
    ],
    dtype=np.float64,
)

POSE_FROM_FEATURE_RIGHT = np.asarray(
    [
        [0.0000803453, 0.0015914161, -0.0057728689],
        [0.0011396475, -0.0004601033, -0.0285922273],
        [0.0052445990, -0.0022322005, 1.5984186200],
    ],
    dtype=np.float64,
)

# 성공 허용 범위. calibration의 in-sample 오차를 고려해 과도한 미세수정을 막는다.
X_TOL_M = 0.010
Y_TOL_M = 0.010
YAW_TOL_DEG = 0.40

# 첫 보정 + 필요할 때 한 번의 추가 보정.
MAX_CORRECTIONS = 5

# 3축 비율을 보존한 채 한 명령 전체를 함께 축소한다.
MAX_FORWARD_M = 0.070
MAX_BACKWARD_M = 0.080
MAX_LATERAL_M = 0.070
MAX_YAW_DEG = 2.50

# 실제 로봇에서 좌우 보정이 과하게 적용되어 Y 명령만 완화한다.
LATERAL_GAIN = 0.65

# 잘못된 검출값으로 큰 동작을 만들지 않는 hard reject.
HARD_X_M = 0.300
HARD_Y_M = 0.300
HARD_YAW_DEG = 15.0

MEASURE_FRAMES = 24
MIN_TOP_FRAMES = 12
MIN_SIDE_FRAMES = 12
MIN_LOCKED_SIDE_FRAMES = 8
MEASURE_TIMEOUT_S = 3.0
CAMERA_FLUSH_FRAMES = 3
ANGLE_CLUSTER_RADIUS_DEG = 0.45
# recovery 전에는 한 window만 판단하고 바로 3 cm 접근한다.
RECOVERY_TRIGGER_RETRIES = 1
MAX_STATIONARY_RETRIES = 2
RETRY_WAIT_S = 0.1

# 최초 위치에서 corner를 선택하지 못할 때만 사용하는 제한된 recovery.
RECOVERY_FORWARD_M = 0.030
MAX_FORWARD_RECOVERIES = 2

# 양쪽 corner가 서로 다른 물체를 잡았는지 판단하는 한계.
SIDE_DISAGREE_TRANSLATION_M = 0.030
SIDE_DISAGREE_YAW_DEG = 0.50
EXPECTED_SCORE_LIMIT = 3.0
EXPECTED_SCORE_ADVANTAGE = 1.0

SETTLE_S = 0.25
LINEAR_SPEED_MPS = 0.10
ANGULAR_SPEED_RADPS = 0.50
QUINTIC_PEAK = 1.875
MIN_LEG_TIME_S = 0.8

WINDOW_NAME = "Tote Align - New Bracket Fixed Side"


@dataclass(frozen=True)
class Measurement:
    side: str
    feature: np.ndarray
    top_frames: int
    side_frames: int


@dataclass(frozen=True)
class PoseError:
    x_m: float
    y_m: float
    yaw_deg: float


@dataclass(frozen=True)
class RelativeCommand:
    x_m: float
    y_m: float
    yaw_deg: float


def wrap_angle_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def dominant_angle_mask(angles: np.ndarray) -> np.ndarray:
    """가장 큰 TOP angle 군집만 남겨 다른 선 가설이 섞이지 않게 한다."""
    distances = np.abs(angles[:, None] - angles[None, :])
    seed_index = int(np.argmax(np.sum(distances <= ANGLE_CLUSTER_RADIUS_DEG, axis=1)))
    first = np.abs(angles - angles[seed_index]) <= ANGLE_CLUSTER_RADIUS_DEG
    center = float(np.median(angles[first]))
    return np.abs(angles - center) <= ANGLE_CLUSTER_RADIUS_DEG


def pose_vector(side: str, feature: np.ndarray) -> np.ndarray:
    target = TARGET_RIGHT if side == "RIGHT" else TARGET_LEFT
    inverse = POSE_FROM_FEATURE_RIGHT if side == "RIGHT" else POSE_FROM_FEATURE_LEFT
    return inverse @ (feature - target)


def expected_score(pose: np.ndarray, expected: PoseError) -> float:
    """직전 이동으로 예상한 잔여오차와의 정규화 거리."""
    return math.sqrt(
        ((pose[0] - expected.x_m) / 0.020) ** 2
        + ((pose[1] - expected.y_m) / 0.020) ** 2
        + ((pose[2] - expected.yaw_deg) / 0.50) ** 2
    )


def measure_feature(
    pipeline,
    *,
    show: bool,
    label: str,
    expected_error: PoseError | None = None,
    required_side: str | None = None,
) -> Measurement | None:
    """양쪽 추정의 일관성을 확인하고 신뢰할 수 있는 corner를 선택한다."""
    rows: list[tuple[float, float | None, float | None, float | None, float | None]] = []
    reset_top_tracking()
    flush_camera(pipeline, frame_count=CAMERA_FLUSH_FRAMES)
    deadline = time.monotonic() + MEASURE_TIMEOUT_S

    while len(rows) < MEASURE_FRAMES and time.monotonic() < deadline:
        try:
            frames = pipeline.wait_for_frames()
        except RuntimeError as error:
            print(f"{label}: 카메라 frame 오류: {error}")
            continue

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asarray(color_frame.get_data())
        feature = detect_frame_feature(image)

        if show:
            cv2.imshow(WINDOW_NAME, draw_feature(image, feature, label=label))
            cv2.waitKey(1)

        if feature is None:
            continue

        left_x = left_y = right_x = right_y = None
        if feature.left is not None:
            left_x = float(feature.left.point[0])
            left_y = float(feature.left.point[1])
        if feature.right is not None:
            right_x = float(feature.right.point[0])
            right_y = float(feature.right.point[1])

        rows.append(
            (float(feature.top_angle_deg), left_x, left_y, right_x, right_y)
        )

    if len(rows) < MIN_TOP_FRAMES:
        print(f"{label}: TOP 검출 부족 ({len(rows)}/{MIN_TOP_FRAMES})")
        return None

    angles = np.asarray([row[0] for row in rows], dtype=np.float64)
    selected = dominant_angle_mask(angles)
    right_rows = [
        (row[3], row[4], row[0])
        for row, keep in zip(rows, selected)
        if keep and row[3] is not None
    ]
    left_rows = [
        (row[1], row[2], row[0])
        for row, keep in zip(rows, selected)
        if keep and row[1] is not None
    ]

    right_min_frames = (
        MIN_LOCKED_SIDE_FRAMES
        if required_side == "RIGHT"
        else MIN_SIDE_FRAMES
    )
    left_min_frames = (
        MIN_LOCKED_SIDE_FRAMES
        if required_side == "LEFT"
        else MIN_SIDE_FRAMES
    )

    right_feature = (
        np.median(np.asarray(right_rows, dtype=np.float64), axis=0)
        if len(right_rows) >= right_min_frames
        else None
    )
    left_feature = (
        np.median(np.asarray(left_rows, dtype=np.float64), axis=0)
        if len(left_rows) >= left_min_frames
        else None
    )

    if right_feature is None and left_feature is None:
        print(
            f"{label}: corner 검출 부족 "
            f"(TOP={len(rows)}, RIGHT={len(right_rows)}, LEFT={len(left_rows)})"
        )
        return None

    # 첫 보정에서 선택한 side를 뒤의 검증에서도 계속 사용한다.
    # 반대 side만 보이면 전환하지 않고 정지 재측정한다.
    if required_side is not None:
        selected_feature = right_feature if required_side == "RIGHT" else left_feature
        selected_frames = len(right_rows) if required_side == "RIGHT" else len(left_rows)

        if selected_feature is None:
            print(
                f"{label}: 고정한 {required_side} corner 검출 부족 "
                f"(TOP={len(rows)}, RIGHT={len(right_rows)}, LEFT={len(left_rows)})"
            )
            return None

        measurement = Measurement(
            required_side,
            selected_feature,
            len(rows),
            selected_frames,
        )
        print(
            f"{label}: {measurement.side} feature="
            f"({measurement.feature[0]:.3f}, {measurement.feature[1]:.3f}, "
            f"{measurement.feature[2]:+.4f}) | "
            f"TOP={measurement.top_frames}, side={measurement.side_frames}"
        )
        return measurement

    selected_side: str
    selected_feature: np.ndarray
    selected_frames: int

    if right_feature is not None and left_feature is not None:
        right_pose = pose_vector("RIGHT", right_feature)
        left_pose = pose_vector("LEFT", left_feature)
        translation_disagreement = float(np.linalg.norm(right_pose[:2] - left_pose[:2]))
        yaw_disagreement = abs(float(right_pose[2] - left_pose[2]))

        if (
            translation_disagreement <= SIDE_DISAGREE_TRANSLATION_M
            and yaw_disagreement <= SIDE_DISAGREE_YAW_DEG
        ):
            selected_side, selected_feature, selected_frames = (
                "RIGHT",
                right_feature,
                len(right_rows),
            )
        else:
            print(
                f"{label}: LEFT/RIGHT 충돌 "
                f"(translation={translation_disagreement * 100:.2f} cm, "
                f"yaw={yaw_disagreement:.3f} deg, "
                f"frames L={len(left_rows)} R={len(right_rows)})"
            )
            print(
                f"{label}: LEFT pose="
                f"({left_pose[0] * 100:+.2f} cm, {left_pose[1] * 100:+.2f} cm, "
                f"{left_pose[2]:+.3f} deg) | RIGHT pose="
                f"({right_pose[0] * 100:+.2f} cm, {right_pose[1] * 100:+.2f} cm, "
                f"{right_pose[2]:+.3f} deg)"
            )

            if expected_error is not None:
                right_score = expected_score(right_pose, expected_error)
                left_score = expected_score(left_pose, expected_error)
                print(
                    f"{label}: 예상 잔여오차 score "
                    f"LEFT={left_score:.2f}, RIGHT={right_score:.2f}"
                )
                if (
                    left_score <= EXPECTED_SCORE_LIMIT
                    and right_score - left_score >= EXPECTED_SCORE_ADVANTAGE
                ):
                    selected_side, selected_feature, selected_frames = (
                        "LEFT",
                        left_feature,
                        len(left_rows),
                    )
                elif (
                    right_score <= EXPECTED_SCORE_LIMIT
                    and left_score - right_score >= EXPECTED_SCORE_ADVANTAGE
                ):
                    selected_side, selected_feature, selected_frames = (
                        "RIGHT",
                        right_feature,
                        len(right_rows),
                    )
                else:
                    print(f"{label}: 어느 corner가 맞는지 판단할 수 없어 이동하지 않습니다.")
                    return None
            else:
                # 새 calibration에서 RIGHT 모델의 잔차가 훨씬 작았다.
                # LEFT가 뚜렷하게 우세하지 않으면 기존 정책대로 RIGHT를 사용한다.
                selected_side, selected_feature, selected_frames = (
                    "RIGHT",
                    right_feature,
                    len(right_rows),
                )
                print(f"{label}: 정밀한 RIGHT 모델을 선택합니다.")
    elif right_feature is not None:
        selected_side, selected_feature, selected_frames = (
            "RIGHT",
            right_feature,
            len(right_rows),
        )
    else:
        selected_side, selected_feature, selected_frames = (
            "LEFT",
            left_feature,
            len(left_rows),
        )

    measurement = Measurement(
        selected_side,
        selected_feature,
        len(rows),
        selected_frames,
    )

    print(
        f"{label}: {measurement.side} feature="
        f"({measurement.feature[0]:.3f}, {measurement.feature[1]:.3f}, "
        f"{measurement.feature[2]:+.4f}) | "
        f"TOP={measurement.top_frames}, side={measurement.side_frames}"
    )
    return measurement


def estimate_pose_error(measurement: Measurement) -> PoseError:
    pose = pose_vector(measurement.side, measurement.feature)
    error = PoseError(float(pose[0]), float(pose[1]), float(pose[2]))
    print(
        f"현재 pose error [{measurement.side}]: "
        f"x={error.x_m * 100:+.2f} cm, "
        f"y={error.y_m * 100:+.2f} cm, yaw={error.yaw_deg:+.3f} deg"
    )
    return error


def within_tolerance(error: PoseError) -> bool:
    return (
        abs(error.x_m) <= X_TOL_M
        and abs(error.y_m) <= Y_TOL_M
        and abs(error.yaw_deg) <= YAW_TOL_DEG
    )


def pose_error_to_command(error: PoseError) -> RelativeCommand:
    """목표 frame의 pose 오차를 현재 robot body frame의 역이동으로 바꾼다."""
    theta = math.radians(error.yaw_deg)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    raw_y_m = +sin_theta * error.x_m - cos_theta * error.y_m
    return RelativeCommand(
        x_m=-cos_theta * error.x_m - sin_theta * error.y_m,
        y_m=LATERAL_GAIN * raw_y_m,
        yaw_deg=-error.yaw_deg,
    )


def residual_after_command(error: PoseError, command: RelativeCommand) -> PoseError:
    """현재 pose에 body-frame 명령을 합성해 다음 측정의 예상 잔여오차를 구한다."""
    theta = math.radians(error.yaw_deg)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return PoseError(
        x_m=error.x_m + cos_theta * command.x_m - sin_theta * command.y_m,
        y_m=error.y_m + sin_theta * command.x_m + cos_theta * command.y_m,
        yaw_deg=error.yaw_deg + command.yaw_deg,
    )


def limit_command(command: RelativeCommand) -> tuple[RelativeCommand, float]:
    """x/y/yaw 결합 방향을 유지하면서 안전 범위로 공통 scale한다."""
    scales = [1.0]
    x_limit = MAX_FORWARD_M if command.x_m >= 0.0 else MAX_BACKWARD_M
    if abs(command.x_m) > x_limit:
        scales.append(x_limit / abs(command.x_m))
    if abs(command.y_m) > MAX_LATERAL_M:
        scales.append(MAX_LATERAL_M / abs(command.y_m))
    if abs(command.yaw_deg) > MAX_YAW_DEG:
        scales.append(MAX_YAW_DEG / abs(command.yaw_deg))

    scale = min(scales)
    return (
        RelativeCommand(
            command.x_m * scale,
            command.y_m * scale,
            command.yaw_deg * scale,
        ),
        scale,
    )


def trajectory_duration(command: RelativeCommand) -> float:
    translation = math.hypot(command.x_m, command.y_m)
    linear_time = QUINTIC_PEAK * translation / LINEAR_SPEED_MPS
    angular_time = QUINTIC_PEAK * abs(math.radians(command.yaw_deg)) / ANGULAR_SPEED_RADPS
    return max(MIN_LEG_TIME_S, linear_time, angular_time)


def offset_world_pose(reference, command: RelativeCommand) -> tuple[float, float, float]:
    ref_x, ref_y, ref_yaw = reference
    cos_yaw = math.cos(ref_yaw)
    sin_yaw = math.sin(ref_yaw)
    return (
        ref_x + cos_yaw * command.x_m - sin_yaw * command.y_m,
        ref_y + sin_yaw * command.x_m + cos_yaw * command.y_m,
        ref_yaw + math.radians(command.yaw_deg),
    )


def command_to_world_pose(current, target) -> RelativeCommand:
    cur_x, cur_y, cur_yaw = current
    target_x, target_y, target_yaw = target
    dx = target_x - cur_x
    dy = target_y - cur_y
    cos_yaw = math.cos(cur_yaw)
    sin_yaw = math.sin(cur_yaw)
    return RelativeCommand(
        x_m=cos_yaw * dx + sin_yaw * dy,
        y_m=-sin_yaw * dx + cos_yaw * dy,
        yaw_deg=math.degrees(wrap_angle_rad(target_yaw - cur_yaw)),
    )


def send_coupled_command(
    robot,
    monitor: OdometryMonitor,
    command: RelativeCommand,
    *,
    label: str,
) -> bool:
    """결합 명령을 보내며 stream 만료 시 동일 world 목표의 잔여량만 재시도한다."""
    start = odom_pose(monitor.odom)
    world_target = offset_world_pose(start, command)
    pending = command

    for attempt in range(2):
        if attempt:
            pending = command_to_world_pose(odom_pose(monitor.odom), world_target)

        print(
            f"{label} ({attempt + 1}/2): x={pending.x_m:+.4f} m, "
            f"y={pending.y_m:+.4f} m, yaw={pending.yaw_deg:+.3f} deg"
        )
        leg = build_leg(
            start=odom_pose(monitor.odom),
            target=(pending.x_m, pending.y_m, math.radians(pending.yaw_deg)),
            absolute=False,
            duration=trajectory_duration(pending),
            turn_direction="shortest",
        )
        try:
            return bool(move_leg(robot, monitor, leg, settle=SETTLE_S))
        except RuntimeError as error:
            if "stream is expired" not in str(error).lower() or attempt == 1:
                raise
            print("command stream 만료: 현재 odometry에서 남은 이동을 한 번 재전송합니다.")

    return False


class ToteAligner:
    """양쪽 corner 선택과 방향별 recovery를 사용하는 tote aligner."""

    def __init__(self, camera_serial: str | None = None, show: bool = False, *, camera=None):
        if camera is None and not camera_serial:
            raise ValueError("camera_serial 또는 공용 camera 중 하나가 필요합니다.")

        self.camera_serial = camera_serial
        self.show = show
        self._camera = camera
        self._pipeline = None

    @property
    def pipeline(self):
        if self._camera is not None:
            return self._camera.pipeline
        return self._pipeline

    @property
    def started(self) -> bool:
        return self.pipeline is not None

    def start(self) -> None:
        if self.started:
            return

        if self._camera is not None:
            raise RuntimeError("공용 camera pipeline이 시작되지 않았습니다.")

        self._pipeline = start_camera(self.camera_serial)

    def stop(self) -> None:
        if self._camera is None and self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

        if self.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    def _measure_with_stationary_retry(
        self,
        label: str,
        expected_error: PoseError | None,
        required_side: str | None,
        retries: int,
    ) -> Measurement | None:
        for attempt in range(retries):
            measurement = measure_feature(
                self.pipeline,
                show=self.show,
                label=f"{label} {attempt + 1}/{retries}",
                expected_error=expected_error,
                required_side=required_side,
            )
            if measurement is not None:
                return measurement
            time.sleep(RETRY_WAIT_S)
        return None

    def align(
        self,
        robot,
        monitor: OdometryMonitor,
        *,
        verify: bool = True,
        abort_check=None,
    ) -> bool:
    
        """최대 두 번의 결합 보정으로 최종 파지 기준에 정렬한다."""
        expected_error = None
        active_side: str | None = None
        correction_index = 0
        recovery_count = 0

        while correction_index <= MAX_CORRECTIONS:
            measure_retries = (
                RECOVERY_TRIGGER_RETRIES
                if correction_index == 0
                and recovery_count < MAX_FORWARD_RECOVERIES
                else MAX_STATIONARY_RETRIES
            )
            measurement = self._measure_with_stationary_retry(
                f"측정 {correction_index + 1}",
                expected_error,
                active_side,
                measure_retries,
            )
            if measurement is None:
                # 아직 정상 측정/보정을 한 번도 하지 못한 경우에만 앞으로 3 cm 접근한다.
                # 이미 보정한 뒤라면 tote에 가까울 수 있으므로 recovery 전진을 금지한다.
                if correction_index == 0 and recovery_count < MAX_FORWARD_RECOVERIES:
                    recovery_count += 1
                    recovery_command = RelativeCommand(
                        x_m=RECOVERY_FORWARD_M,
                        y_m=0.0,
                        yaw_deg=0.0,
                    )
                    print(
                        f"전방 recovery {recovery_count}/{MAX_FORWARD_RECOVERIES}: "
                        f"신뢰 가능한 corner 없음 -> 앞으로 {RECOVERY_FORWARD_M * 100:.1f} cm"
                    )
                    if not send_coupled_command(
                        robot,
                        monitor,
                        recovery_command,
                        label="전방 recovery",
                    ):
                        print("Tote 정렬 실패: 전방 recovery 이동이 완료되지 않았습니다.")
                        return True

                    expected_error = None
                    continue

                if active_side is None:
                    print(
                        "Tote 정렬 실패: 전방 recovery 후에도 "
                        "corner를 선택하지 못했습니다."
                    )
                else:
                    print(
                        f"Tote 정렬 실패: 고정한 {active_side} corner를 "
                        "정지 재측정해도 충분히 검출하지 못했습니다."
                    )
                return True

            if active_side is None:
                active_side = measurement.side
                print(f"정렬 기준 side 고정: {active_side}")

            error = estimate_pose_error(measurement)
            if within_tolerance(error):
                print(
                    "Tote 정렬 성공: "
                    f"|x|<={X_TOL_M * 100:.1f} cm, "
                    f"|y|<={Y_TOL_M * 100:.1f} cm, "
                    f"|yaw|<={YAW_TOL_DEG:.2f} deg"
                )
                return True

            if correction_index == MAX_CORRECTIONS:
                print("Tote 정렬 실패: 두 번 보정 후에도 허용 범위 밖입니다.")
                return True

            if (
                abs(error.x_m) > HARD_X_M
                or abs(error.y_m) > HARD_Y_M
                or abs(error.yaw_deg) > HARD_YAW_DEG
            ):
                print("Tote 정렬 실패: 추정 오차가 hard limit를 넘었습니다. 이동하지 않습니다.")
                return True

            raw_command = pose_error_to_command(error)
            command, scale = limit_command(raw_command)
            expected_error = residual_after_command(error, command)
            print(
                f"결합 보정 {correction_index + 1}/{MAX_CORRECTIONS}: "
                f"x={command.x_m * 100:+.2f} cm, "
                f"y={command.y_m * 100:+.2f} cm, "
                f"yaw={command.yaw_deg:+.3f} deg, scale={scale:.3f}"
            )
            if not send_coupled_command(
                robot,
                monitor,
                command,
                label=f"결합 보정 {correction_index + 1}",
            ):
                print("Tote 정렬 실패: base 이동이 완료되지 않았습니다.")
                return True

            correction_index += 1

        return True


def align_tote(
    robot,
    monitor: OdometryMonitor,
    *,
    camera_serial: str | None = None,
    show: bool = False,
    camera=None,
    pipeline=None,
) -> bool:
    """데모용 wrapper. 기존 camera 또는 pipeline을 전달하면 소유권을 건드리지 않는다."""
    own_camera = None
    if pipeline is None and camera is not None:
        pipeline = camera.pipeline
    if pipeline is None:
        own_camera = RealSenseCamera(640, 480, 30, serial=camera_serial)
        own_camera.start()
        pipeline = own_camera.pipeline

    try:
        return ToteAligner(pipeline, show=show).align(robot, monitor)
    finally:
        if own_camera is not None:
            own_camera.stop()
        if show:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except cv2.error:
                pass


def main() -> None:
    # standalone 테스트에만 필요한 값은 모두 main 안에 둔다.
    address = "192.168.30.1:50051"
    camera_serial = "250122079439"
    camera_width = 640
    camera_height = 480
    camera_fps = 30

    initial_torso = np.deg2rad([0.0, 30.0, -50.0, 30.0, 0.0, 0.0]).tolist()
    head_down = np.deg2rad([0.0, 43.0]).tolist()
    before_right = np.deg2rad(
        [-38.23, -53.19, -21.31, -48.14, -63.73, 81.18, 2.39]
    ).tolist()
    before_left = np.deg2rad(
        [-38.23, 53.19, 21.31, -48.14, 63.73, 81.18, -2.39]
    ).tolist()

    parser = argparse.ArgumentParser(
        description="New bracket coupled tote align with fixed measurement side"
    )
    parser.add_argument("--address", default=address)
    parser.add_argument("--model", choices=("a", "m"), default="m")
    parser.add_argument("--camera-serial", default=camera_serial)
    parser.add_argument("--show-tote", action="store_true")
    args = parser.parse_args()

    robot = initialize_mobile(
        args.address,
        args.model,
        power=".*",
        servo=".*",
        unlimited=False,
    )
    monitor = OdometryMonitor()
    camera = RealSenseCamera(
        camera_width,
        camera_height,
        camera_fps,
        serial=args.camera_serial,
    )
    state_update_started = False
    camera_started = False

    try:
        robot.start_state_update(monitor.on_state, rate=50)
        state_update_started = True
        if not wait_for_odometry(monitor):
            raise RuntimeError("Odometry를 받지 못했습니다.")

        print("초기 Torso / Head 및 양팔 BEFORE 자세로 동시에 이동")
        with ThreadPoolExecutor(max_workers=2) as executor:
            torso_future = executor.submit(
                move_torso_and_head,
                robot,
                initial_torso,
                head_down,
                minimum_time=1.5,
            )
            arms_future = executor.submit(
                move_both_arms,
                robot,
                before_right,
                before_left,
                minimum_time=1.5,
            )
            torso_ok = torso_future.result()
            arms_ok = arms_future.result()

        if not torso_ok or not arms_ok:
            raise RuntimeError("초기 Torso / Head / BEFORE 자세 이동 실패")

        camera.start()
        camera_started = True
        success = ToteAligner(camera.pipeline, show=args.show_tote).align(robot, monitor)
        print(f"최종 결과: {'SUCCESS' if success else 'FAIL'}")

    except KeyboardInterrupt:
        print("\n사용자가 중단했습니다. 로봇은 현재 위치에서 종료합니다.")
    finally:
        if camera_started:
            camera.stop()
        if state_update_started:
            robot.stop_state_update()
        try:
            robot.disable_control_manager()
        except Exception:
            pass
        robot.disconnect()
        if args.show_tote:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        print("모든 연결과 제어를 정리했습니다.")


if __name__ == "__main__":
    main()
