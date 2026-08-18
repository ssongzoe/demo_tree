"""RB-Y1 양팔을 before-grasp 자세로 이동한 뒤 grasp 자세로 이동하는 실습 코드."""

import argparse

import numpy as np
import rby1_sdk as rby


BEFORE_GRASP_RIGHT = np.deg2rad([-33.0, -43.0, -24.0, -84.0, -60.0, 87.0, -13.0])
BEFORE_GRASP_LEFT = np.deg2rad([-33.0, 43.0, 24.0, -84.0, 60.0, 87.0, 13.0])

GRASP_RIGHT = np.deg2rad([-33.0, -32.0, -13.5, -86.0, -57.0, 87.0, -13.0])
GRASP_LEFT = np.deg2rad([-33.0, 32.0, 13.5, -86.0, 57.0, 87.0, 13.0])


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

    # 실습 편의를 위해 전원과 서보를 켜고 제어 관리자를 활성화한다.
    if not robot.power_on(".*"):
        raise RuntimeError("로봇 전원을 켜지 못했습니다.")
    if not robot.servo_on(".*"):
        raise RuntimeError("서보를 켜지 못했습니다.")

    robot.reset_fault_control_manager()
    if not robot.enable_control_manager(unlimited_mode_enabled=True):
        raise RuntimeError("제어 관리자를 활성화하지 못했습니다.")

    print("[1/2] Before-grasp 자세로 이동합니다. (5초)")
    move_both_arms(robot, BEFORE_GRASP_RIGHT, BEFORE_GRASP_LEFT, minimum_time=5.0)

    print("[2/2] Grasp 자세로 이동합니다. (3초)")
    move_both_arms(robot, GRASP_RIGHT, GRASP_LEFT, minimum_time=3.0)

    print("Grasp 자세 이동 완료")


if __name__ == "__main__":
    main()