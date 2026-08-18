"""RB-Y1 양팔을 before-grasp 자세로 이동한 뒤 grasp 자세로 이동하는 실습 코드."""

import argparse

import numpy as np
import rby1_sdk as rby


BEFORE_GRASP_RIGHT = np.deg2rad([-31.0, -54.0, -10.0, -91.0, -57.0, 87.0, -13.0])
BEFORE_GRASP_LEFT = np.deg2rad([-31.0, 54.0, 10.0, -91.0, 57.0, 87.0, 13.0])

GRASP_RIGHT = np.deg2rad([-31.0, -43.5, -9.0, -91.0, -57.0, 87.0, -13.0])
GRASP_LEFT = np.deg2rad([-31.0, 43.5, 9.0, -91.0, 57.0, 87.0, 13.0])


UP_RIGHT = np.deg2rad([-31.0, -43.5, -17.0, -89.0, -65.0, 87.0, -13.0])
UP_LEFT = np.deg2rad([-31.0, 43.5, 17.0, -89.0, 65.0, 87.0, 13.0])


def create_robot(address: str, model: str):
    """선택한 RB-Y1 모델의 로봇 객체를 생성한다."""
    if model == "a":
        return rby.create_robot_a(address)
    return rby.create_robot_m(address)


def move_both_arms(robot, right_position: np.ndarray, left_position: np.ndarray, minimum_time: float):
    """양팔 관절 목표를 동시에 전송하고 동작이 끝날 때까지 기다린다."""
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
    parser = argparse.ArgumentParser(description="RB-Y1 양팔 grasp 자세 실습")
    parser.add_argument("--address", default="192.168.30.1:50051", help="로봇 주소")
    parser.add_argument("--model", choices=("a", "m"), default="m", help="RB-Y1 모델")
    args = parser.parse_args()

    robot = create_robot(args.address, args.model)
    if not robot.connect():
        raise RuntimeError(f"로봇 연결 실패: {args.address}")



    # 전원이 꺼져 있을 때만 전원을 켠다.
    if robot.is_power_on(".*"):
        print("Power: 이미 켜져 있음")
    else:
        power_result = robot.power_on(".*")
        print(f"power_on 결과: {power_result}")

    # 서보가 꺼져 있을 때만 서보를 켠다.
    if robot.is_servo_on(".*"):
        print("Servo: 이미 켜져 있음")
    else:
        servo_result = robot.servo_on(".*")
        print(f"servo_on 결과: {servo_result}")

    # Control Manager에 fault가 있을 때만 초기화한다.
    cm_state = robot.get_control_manager_state()
    print(f"Control Manager 상태: {cm_state.state}")

    if cm_state.state in (
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ):
        print("Control Manager fault를 초기화합니다.")
        reset_result = robot.reset_fault_control_manager()
        print(f"reset_fault_control_manager 결과: {reset_result}")

    # 현재 코드는 Joint Position 제어이므로 unlimited mode가 필요하지 않다.
    enable_result = robot.enable_control_manager(
        unlimited_mode_enabled=False,
    )
    print(f"enable_control_manager 결과: {enable_result}")


    print("[1/2] Before-grasp 자세로 이동합니다. (3초)")
    move_both_arms(robot, BEFORE_GRASP_RIGHT, BEFORE_GRASP_LEFT, minimum_time=3.0)

    print("[2/2] Grasp 자세로 이동합니다. (3초)")
    move_both_arms(robot, GRASP_RIGHT, GRASP_LEFT, minimum_time=3.0)

    print("Grasp 자세 이동 완료")


    move_both_arms(robot, UP_RIGHT, UP_LEFT, minimum_time=1.0)

    print("box up")


if __name__ == "__main__":
    main()