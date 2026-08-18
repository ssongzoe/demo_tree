#!/usr/bin/env python3
"""RB-Y1 양팔 파지 실습.

Before-grasp → Grasp → 저토크 그리퍼 닫기 → 사용자 확인 → Box-up 순서로 실행한다.
"""

import argparse
import time

import numpy as np
import rby1_sdk as rby

from control.gripper_controller import GripperController


BEFORE_GRASP_RIGHT = np.deg2rad([-31.0, -54.0, -10.0, -91.0, -57.0, 87.0, -13.0])
BEFORE_GRASP_LEFT = np.deg2rad([-31.0, 54.0, 10.0, -91.0, 57.0, 87.0, 13.0])

GRASP_RIGHT = np.deg2rad([-31.0, -43.5, -9.0, -91.0, -57.0, 87.0, -13.0])
GRASP_LEFT = np.deg2rad([-31.0, 43.5, 9.0, -91.0, 57.0, 87.0, 13.0])

UP_RIGHT = np.deg2rad([-31.0, -43.5, -17.0, -89.0, -65.0, 87.0, -13.0])
UP_LEFT = np.deg2rad([-31.0, 43.5, 17.0, -89.0, 65.0, 87.0, 13.0])

DEFAULT_GRIPPER_TARGET = 0.35
DEFAULT_GRIPPER_TORQUE = 0.20


def create_robot(address: str, model: str):
    """선택한 RB-Y1 모델의 로봇 객체를 생성한다."""
    if model == "a":
        return rby.create_robot_a(address)
    return rby.create_robot_m(address)


def prepare_robot(robot) -> bool:
    """전원, 서보, fault와 Control Manager 상태를 준비한다."""
    if robot.is_power_on(".*"):
        print("Power: 이미 켜져 있음")
    else:
        power_result = robot.power_on(".*")
        print(f"power_on 결과: {power_result}")
        if not power_result:
            return False

    if robot.is_servo_on(".*"):
        print("Servo: 이미 켜져 있음")
    else:
        servo_result = robot.servo_on(".*")
        print(f"servo_on 결과: {servo_result}")
        if not servo_result:
            return False

    cm_state = robot.get_control_manager_state()
    print(f"Control Manager 상태: {cm_state.state}")

    if cm_state.state in (
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ):
        print("Control Manager fault를 초기화합니다.")
        reset_result = robot.reset_fault_control_manager()
        print(f"reset_fault_control_manager 결과: {reset_result}")

    enable_result = robot.enable_control_manager(unlimited_mode_enabled=False)
    print(f"enable_control_manager 결과: {enable_result}")
    return bool(enable_result)


def move_both_arms(
    robot,
    right_position: np.ndarray,
    left_position: np.ndarray,
    minimum_time: float,
):
    """양팔 관절 목표를 동시에 보내고 동작이 끝날 때까지 기다린다."""
    command = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_right_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(right_position)
            )
            .set_left_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(left_position)
            )
        )
    )

    feedback = robot.send_command(command).get()
    print(f"동작 완료: {feedback.finish_code}")
    return feedback.finish_code


def main() -> None:
    parser = argparse.ArgumentParser(description="RB-Y1 양팔 grasp 및 box-up 실습")
    parser.add_argument("--address", default="192.168.30.1:50051", help="로봇 주소")
    parser.add_argument("--model", choices=("a", "m"), default="m", help="RB-Y1 모델")
    parser.add_argument(
        "--gripper-target",
        type=float,
        default=DEFAULT_GRIPPER_TARGET,
        help="그리퍼 닫힘 위치, 0.0=열림, 1.0=완전 닫힘",
    )
    parser.add_argument(
        "--gripper-torque",
        type=float,
        default=DEFAULT_GRIPPER_TORQUE,
        help="그리퍼 파지 토크 [Nm], 최대 0.46",
    )
    args = parser.parse_args()

    robot = create_robot(args.address, args.model)
    gripper = None

    if not robot.connect():
        print(f"로봇 연결 실패: {args.address}")
        return

    if not prepare_robot(robot):
        print("로봇 제어 준비 실패")
        return

    robot.set_tool_flange_output_voltage("right", 12)
    robot.set_tool_flange_output_voltage("left", 12)
    time.sleep(0.5)


    gripper = GripperController(position_torque=args.gripper_torque)
    gripper.connect()
    gripper.open(duration=2.0)

    print("[1/4] Before-grasp 자세로 이동합니다. (3초)")
    move_both_arms(robot, BEFORE_GRASP_RIGHT, BEFORE_GRASP_LEFT, minimum_time=1.5)

    print("[2/4] Grasp 자세로 이동합니다. (3초)")
    move_both_arms(robot, GRASP_RIGHT, GRASP_LEFT, minimum_time=1.0)

    print(
        f"[3/4] 그리퍼를 닫습니다. "
        f"target={args.gripper_target:.2f}, torque={args.gripper_torque:.2f} Nm"
    )
    gripper.close(
        target=args.gripper_target,
        torque=args.gripper_torque,
        duration=1.0,
    )
    print(f"그리퍼 현재 위치: {gripper.get_positions().round(3)}")

    print("[4/4] 박스를 들어 올립니다. (2초)")
    move_both_arms(robot, UP_RIGHT, UP_LEFT, minimum_time=2.0)
    print("Box-up 완료")



if __name__ == "__main__":
    main()