#!/usr/bin/env python3
"""RB-Y1 상체 제어 모듈.

rby1_sdk에서 사용하는 주요 제어
- JointPositionCommandBuilder: 관절 목표 위치 명령 생성
- BodyComponentBasedCommandBuilder: torso / 양팔 명령 구성
- HeadCommandBuilder: head 명령 구성
- RobotCommandBuilder: 최종 로봇 명령 생성
- robot.send_command(): 상체 관절 명령 전송

※ 로봇 연결 및 모바일 베이스 초기화는 mobile_controller.py에서 담당한다.
"""

import numpy as np
import rby1_sdk as rby


READY_POSE = {
    "torso": np.deg2rad([0.0, 20.0, -45.0, 45.0, 0.0, 0.0]).tolist(),
    "right_arm": np.deg2rad([-10.0, -40.0, -5.0, -110.0, -35.0, 50.0, 0.0]).tolist(),
    "left_arm": np.deg2rad([-10.0, 40.0, -5.0, -110.0, 35.0, 50.0, 0.0]).tolist(),
    "head": np.deg2rad([0.0, 43.0]).tolist(),
}

READY_MINIMUM_TIME = 2.0
READY_TIMEOUT_MS = 20000


def _joint_position_command(position, minimum_time):
    """하나의 관절 그룹에 사용할 Joint Position 명령을 만든다."""
    return (
        rby.JointPositionCommandBuilder()
        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0.0))
        .set_minimum_time(minimum_time)
        .set_position(position)
    )


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

    return move_to_upper_body_pose(
        robot,
        READY_POSE,
        minimum_time=READY_MINIMUM_TIME,
        timeout_ms=READY_TIMEOUT_MS,
    )


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