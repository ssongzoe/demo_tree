#!/usr/bin/env python3
"""Tote 양팔 grasp skill.

Before-grasp → Grasp → 그리퍼 닫기 → Box-up을 수행한다.
Pose 값은 기존 demo_grasp.py에서 실제 파지 성공한 값을 그대로 사용한다.
"""

import numpy as np

from control.robot_controller import move_both_arms


BEFORE_GRASP_RIGHT = np.deg2rad([-31.0, -54.0, -10.0, -91.0, -57.0, 87.0, -13.0])
BEFORE_GRASP_LEFT = np.deg2rad([-31.0, 54.0, 10.0, -91.0, 57.0, 87.0, 13.0])

GRASP_RIGHT = np.deg2rad([-31.0, -43.5, -9.0, -91.0, -57.0, 87.0, -13.0])
GRASP_LEFT = np.deg2rad([-31.0, 43.5, 9.0, -91.0, 57.0, 87.0, 13.0])

UP_RIGHT = np.deg2rad([-31.0, -43.5, -17.0, -89.0, -65.0, 87.0, -13.0])
UP_LEFT = np.deg2rad([-31.0, 43.5, 17.0, -89.0, 65.0, 87.0, 13.0])

DEFAULT_GRIPPER_TARGET = 0.35
DEFAULT_GRIPPER_TORQUE = 0.20


def grasp_and_lift(
    robot,
    gripper,
    *,
    gripper_target: float = DEFAULT_GRIPPER_TARGET,
    gripper_torque: float = DEFAULT_GRIPPER_TORQUE,
) -> bool:
    """검증된 양팔 pose로 tote를 파지하고 들어 올린다."""
    print("[Grasp 1/4] Before-grasp 자세")
    if not move_both_arms(robot, BEFORE_GRASP_RIGHT, BEFORE_GRASP_LEFT, minimum_time=1.5):
        return False

    print("[Grasp 2/4] Grasp 자세")
    if not move_both_arms(robot, GRASP_RIGHT, GRASP_LEFT, minimum_time=1.0):
        return False

    print(f"[Grasp 3/4] 그리퍼 닫기: target={gripper_target:.2f}, torque={gripper_torque:.2f} Nm")
    gripper.close(target=gripper_target, torque=gripper_torque, duration=1.0)
    print(f"그리퍼 현재 위치: {gripper.get_positions().round(3)}")

    print("[Grasp 4/4] Box-up 자세")
    if not move_both_arms(robot, UP_RIGHT, UP_LEFT, minimum_time=2.0):
        return False

    return True
