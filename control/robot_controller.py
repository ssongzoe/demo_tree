#!/usr/bin/env python3
"""RB-Y1 상체 제어 모듈."""

import time

import numpy as np
import rby1_sdk as rby


READY_POSE = {
    "torso": np.deg2rad([0.0, 20.0, -45.0, 45.0, 0.0, 0.0]).tolist(),
    "right_arm": np.deg2rad([-10.0, -40.0, -5.0, -110.0, -35.0, 50.0, 0.0]).tolist(),
    "left_arm": np.deg2rad([-10.0, 40.0, -5.0, -110.0, 35.0, 50.0, 0.0]).tolist(),
    "head": np.deg2rad([0.0, 45.0]).tolist(),
}

READY_MINIMUM_TIME = 2.0
READY_TIMEOUT_MS = 20000


def _joint_position_command(position, minimum_time, control_hold_time=0.0):
    """하나의 관절 그룹에 사용할 Joint Position 명령을 만든다."""
    return (
        rby.JointPositionCommandBuilder()
        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time))
        .set_minimum_time(minimum_time)
        .set_position(position)
    )


def _build_torso_and_arms_command(
    torso_position,
    right_position,
    left_position,
    minimum_time,
    control_hold_time=0.0,
):
    """Head는 건드리지 않고 torso와 양팔 목표를 하나의 명령으로 묶는다."""
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_torso_command(_joint_position_command(torso_position, minimum_time, control_hold_time))
            .set_right_arm_command(_joint_position_command(right_position, minimum_time, control_hold_time))
            .set_left_arm_command(_joint_position_command(left_position, minimum_time, control_hold_time))
        )
    )


def _torso_and_arms_vector(pose):
    """torso와 양팔 자세를 하나의 20차원 벡터로 만든다."""
    return np.concatenate((pose["torso"], pose["right_arm"], pose["left_arm"]))


def _smootherstep(progress):
    """양 끝에서 속도와 가속도가 0이 되는 5차 보간 계수를 반환한다."""
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress**3 * (progress * (progress * 6.0 - 15.0) + 10.0)


def build_upper_body_command(pose, minimum_time=2.0):
    """torso, 양팔, head의 Joint Position 명령을 하나로 묶는다."""
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder()
        .set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_torso_command(_joint_position_command(pose["torso"], minimum_time))
            .set_right_arm_command(_joint_position_command(pose["right_arm"], minimum_time))
            .set_left_arm_command(_joint_position_command(pose["left_arm"], minimum_time))
        )
        .set_head_command(rby.HeadCommandBuilder(_joint_position_command(pose["head"], minimum_time)))
    )


def move_to_upper_body_pose(robot, pose, minimum_time=2.0, timeout_ms=20000) -> bool:
    """지정한 상체 관절 자세로 이동한다."""
    handler = robot.send_command(build_upper_body_command(pose, minimum_time=minimum_time))

    if handler.wait_for(timeout_ms) is False:
        handler.cancel()
        handler.wait_for(2000)
        return False

    feedback = handler.get()
    return feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok


def move_to_ready_pose(robot) -> bool:
    """미리 정의된 READY 자세로 이동한다."""
    print("READY 자세로 이동")
    return move_to_upper_body_pose(robot, READY_POSE, minimum_time=READY_MINIMUM_TIME, timeout_ms=READY_TIMEOUT_MS)


def move_both_arms(
    robot,
    right_position: np.ndarray,
    left_position: np.ndarray,
    minimum_time: float,
    timeout_ms: int = 20000,
) -> bool:
    """torso와 head는 유지하고 양팔 관절 목표만 동시에 보낸다."""
    command = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_right_arm_command(_joint_position_command(right_position, minimum_time))
            .set_left_arm_command(_joint_position_command(left_position, minimum_time))
        )
    )

    handler = robot.send_command(command)

    if handler.wait_for(timeout_ms) is False:
        handler.cancel()
        handler.wait_for(2000)
        return False

    feedback = handler.get()
    print(f"양팔 동작 완료: {feedback.finish_code}")
    return feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok


def move_torso_and_arms_through_waypoint(
    robot,
    start_pose,
    waypoint_pose,
    target_pose,
    lift_duration: float,
    pull_duration: float,
    overlap_duration: float = 0.15,
    stream_rate_hz: float = 100.0,
) -> bool:
    """UP 근처부터 PULL 움직임을 겹쳐 관절 공간에서 정지 없이 연결한다."""

    start = _torso_and_arms_vector(start_pose)
    waypoint = _torso_and_arms_vector(waypoint_pose)
    target = _torso_and_arms_vector(target_pose)

    period = 1.0 / stream_rate_hz
    command_horizon = 0.03
    control_hold_time = 0.20
    pull_start_time = lift_duration - overlap_duration
    total_duration = lift_duration + pull_duration
    pull_profile_duration = total_duration - pull_start_time
    stream = robot.create_command_stream(priority=10)

    try:
        started_at = time.monotonic()
        next_send_at = started_at

        while True:
            now = time.monotonic()
            sample_time = min(now - started_at + command_horizon, total_duration)
            lift_ratio = _smootherstep(sample_time / lift_duration)
            pull_ratio = _smootherstep((sample_time - pull_start_time) / pull_profile_duration)
            position = start + lift_ratio * (waypoint - start) + pull_ratio * (target - waypoint)

            stream.send_command(
                _build_torso_and_arms_command(
                    position[:6],
                    position[6:13],
                    position[13:20],
                    minimum_time=command_horizon,
                    control_hold_time=control_hold_time,
                )
            )

            if sample_time >= total_duration:
                # 마지막 PULL target이 적용되기 전에 stream이 취소되지 않도록 잠깐 기다린다.
                time.sleep(0.05)
                break

            next_send_at += period
            sleep_time = next_send_at - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_send_at = time.monotonic()

    except Exception as error:
        print(f"Torso / 양팔 waypoint stream 예외: {error}")
        return False

    finally:
        stream.cancel()
        stream.wait_for(1000)

    print("Torso / 양팔 waypoint stream 완료")
    return True


def move_torso_and_head(
    robot,
    torso_position: np.ndarray,
    head_position: np.ndarray,
    minimum_time: float = 2.0,
    timeout_ms: int = 20000,
) -> bool:
    """양팔은 유지하고 torso와 head 목표만 동시에 보내 perception 기준 자세를 맞춘다."""
    command = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder()
        .set_body_command(
            rby.BodyComponentBasedCommandBuilder().set_torso_command(
                _joint_position_command(torso_position, minimum_time)
            )
        )
        .set_head_command(rby.HeadCommandBuilder(_joint_position_command(head_position, minimum_time)))
    )

    handler = robot.send_command(command)

    if handler.wait_for(timeout_ms) is False:
        handler.cancel()
        handler.wait_for(2000)
        return False

    feedback = handler.get()
    print(f"Torso / Head 동작 완료: {feedback.finish_code}")
    return feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok