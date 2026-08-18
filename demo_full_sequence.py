#!/usr/bin/env python3
"""RB-Y1 박스 파지, 모바일 이송, AR 정렬 및 배치 통합 데모.

동작 순서
1. Before-grasp → Grasp → 그리퍼 닫기 → Box-up
2. 0.10 m 후진 → 회전하며 이동 → 직진
3. AR 마커 기준 yaw 및 위치 정렬
4. UP → Grasp → 그리퍼 열기 → Before-grasp
5. 0.20 m 후진 → 출발 때와 반대 방향으로 회전 → 0.80 m 직진
6. 모든 연결과 제어 종료
"""

import argparse
import math
import time

from control.gripper_controller import GripperController
from control.mobile_controller import (
    OdometryMonitor,
    build_leg,
    move_leg,
    odom_pose,
    wait_for_odometry,
)
from demo_grasp import (
    BEFORE_GRASP_LEFT,
    BEFORE_GRASP_RIGHT,
    GRASP_LEFT,
    GRASP_RIGHT,
    UP_LEFT,
    UP_RIGHT,
    create_robot,
    move_both_arms,
    prepare_robot,
)
from skills.ar_align import align_to_marker


BACK_TARGET = (-0.10, 0.0, 0.0)

TURN_TARGET = (
    -0.50,
    -0.05,
    math.radians(-179.43),
)

STRAIGHT_TARGET = (0.75, 0.0, 0.0)

RETURN_BACK_TARGET = (-0.20, 0.0, 0.0)
RETURN_TURN_TARGET = (0.0, 0.0, math.radians(179.43))
RETURN_STRAIGHT_TARGET = (0.80, 0.0, 0.0)

MARKER_ID = 8
CAMERA_SERIAL = "409122274689"

DEFAULT_GRIPPER_TARGET = 0.20
DEFAULT_GRIPPER_TORQUE = 0.20


def run_mobile_leg(
    robot,
    monitor,
    stream,
    name: str,
    target,
    duration: float,
    stop_at_end: bool,
    settle: float,
) -> bool:
    """현재 odometry 자세를 기준으로 모바일 상대 이동을 한 번 수행한다."""
    leg = build_leg(
        start=odom_pose(monitor.odom),
        target=target,
        absolute=False,
        duration=duration,
        turn_direction="shortest",
    )

    print(name)
    return move_leg(
        robot,
        monitor,
        leg,
        settle=settle,
        stream=stream,
        stop_at_end=stop_at_end,
    )


def run_turn_and_go(robot, monitor) -> bool:
    """0.10 m 후진한 뒤 기존 turn-and-go 경로를 연속 수행한다."""
    stream = robot.create_command_stream(priority=10)

    try:
        if not run_mobile_leg(
            robot,
            monitor,
            stream,
            name="모바일 1/3: 0.10 m 후진",
            target=BACK_TARGET,
            duration=2.0,
            stop_at_end=False,
            settle=0.0,
        ):
            print("0.10 m 후진 실패")
            return False

        if not run_mobile_leg(
            robot,
            monitor,
            stream,
            name="모바일 2/3: 회전 + 이동",
            target=TURN_TARGET,
            duration=7.0,
            stop_at_end=False,
            settle=0.0,
        ):
            print("회전 + 이동 실패")
            return False

        if not run_mobile_leg(
            robot,
            monitor,
            stream,
            name="모바일 3/3: 직진",
            target=STRAIGHT_TARGET,
            duration=5.0,
            stop_at_end=True,
            settle=1.5,
        ):
            print("직진 실패")
            return False

        return True

    finally:
        stream.cancel()
        stream.wait_for(500)


def run_return_route(robot, monitor) -> bool:
    """0.20 m 후진, 왼쪽 회전, 0.80 m 직진으로 복귀 장면을 수행한다."""
    stream = robot.create_command_stream(priority=10)

    try:
        if not run_mobile_leg(
            robot,
            monitor,
            stream,
            name="복귀 1/3: 0.20 m 후진",
            target=RETURN_BACK_TARGET,
            duration=3.0,
            stop_at_end=False,
            settle=0.0,
        ):
            print("복귀 0.20 m 후진 실패")
            return False

        if not run_mobile_leg(
            robot,
            monitor,
            stream,
            name="복귀 2/3: 왼쪽 제자리 회전",
            target=RETURN_TURN_TARGET,
            duration=10.0,
            stop_at_end=False,
            settle=0.0,
        ):
            print("복귀 왼쪽 회전 실패")
            return False

        if not run_mobile_leg(
            robot,
            monitor,
            stream,
            name="복귀 3/3: 0.80 m 직진",
            target=RETURN_STRAIGHT_TARGET,
            duration=5.0,
            stop_at_end=True,
            settle=1.5,
        ):
            print("복귀 0.80 m 직진 실패")
            return False

        return True

    finally:
        stream.cancel()
        stream.wait_for(500)


def grasp_and_lift(robot, gripper, gripper_target: float, gripper_torque: float) -> None:
    """박스 파지 자세로 이동한 뒤 그리퍼를 닫고 Box-up 자세로 들어 올린다."""
    print("[1/9] Before-grasp 자세")
    move_both_arms(robot, BEFORE_GRASP_RIGHT, BEFORE_GRASP_LEFT, minimum_time=1.0)

    print("[2/9] Grasp 자세")
    move_both_arms(robot, GRASP_RIGHT, GRASP_LEFT, minimum_time=1.0)

    print(
        f"[3/9] 그리퍼 닫기: target={gripper_target:.2f}, "
        f"torque={gripper_torque:.2f} Nm"
    )
    gripper.close(
        target=gripper_target,
        torque=gripper_torque,
        duration=1.0,
    )
    print(f"그리퍼 현재 위치: {gripper.get_positions().round(3)}")

    print("[4/9] Box-up 자세")
    move_both_arms(robot, UP_RIGHT, UP_LEFT, minimum_time=1.0)


def lower_release_and_retract(robot, gripper) -> None:
    """UP 자세에서 박스를 내리고 놓은 뒤 Before-grasp 자세로 후퇴한다."""
    print("[7/9] UP → Grasp 자세")
    move_both_arms(robot, GRASP_RIGHT, GRASP_LEFT, minimum_time=1.0)

    print("그리퍼 열기")
    gripper.open(duration=1.0)

    print("[8/9] Grasp → Before-grasp 자세")
    move_both_arms(robot, BEFORE_GRASP_RIGHT, BEFORE_GRASP_LEFT, minimum_time=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="RB-Y1 박스 이송 통합 데모")
    parser.add_argument("--address", default="192.168.30.1:50051", help="로봇 주소")
    parser.add_argument("--model", choices=("a", "m"), default="m", help="RB-Y1 모델")
    parser.add_argument("--marker-id", type=int, default=MARKER_ID, help="정렬에 사용할 마커 ID")
    parser.add_argument("--camera-serial", default=CAMERA_SERIAL, help="AR 정렬용 RealSense 시리얼")
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
    monitor = OdometryMonitor()
    state_update_started = False
    demo_completed = False

    try:
        if not robot.connect():
            raise RuntimeError(f"로봇 연결 실패: {args.address}")

        if not prepare_robot(robot):
            raise RuntimeError("로봇 제어 준비 실패")

        robot.set_tool_flange_output_voltage("right", 12)
        robot.set_tool_flange_output_voltage("left", 12)
        time.sleep(0.5)

        gripper = GripperController(position_torque=args.gripper_torque)
        gripper.connect()
        gripper.open(duration=2.0)

        grasp_and_lift(
            robot,
            gripper,
            gripper_target=args.gripper_target,
            gripper_torque=args.gripper_torque,
        )

        robot.start_state_update(monitor.on_state, rate=50)
        state_update_started = True

        if not wait_for_odometry(monitor):
            raise RuntimeError("Odometry를 받지 못했습니다.")

        print("[5/9] 0.10 m 후진 → Turn-and-go")
        if not run_turn_and_go(robot, monitor):
            raise RuntimeError("Turn-and-go 실패")

        print("[6/9] AR 마커 정렬")
        if not align_to_marker(
            robot,
            monitor,
            marker_id=args.marker_id,
            camera_serial=args.camera_serial,
        ):
            raise RuntimeError("AR 마커 정렬 실패")

        lower_release_and_retract(robot, gripper)

        print("[9/9] 0.20 m 후진 → 왼쪽 회전 → 0.80 m 직진")
        if not run_return_route(robot, monitor):
            raise RuntimeError("복귀 주행 실패")

        demo_completed = True
        print("통합 데모 완료")

    except KeyboardInterrupt:
        print("\n사용자가 통합 데모를 중단했습니다.")

    except Exception as error:
        print(f"통합 데모 실패: {error}")

    finally:
        if state_update_started:
            try:
                robot.stop_state_update()
            except Exception:
                pass

        if gripper is not None:
            gripper.disconnect()

        robot.set_tool_flange_output_voltage("right", 0)
        robot.set_tool_flange_output_voltage("left", 0)
        robot.disable_control_manager()
        robot.disconnect()

        if demo_completed:
            print("모든 연결과 제어를 정상 종료했습니다.")
        else:
            print("오류 종료 후 연결과 제어를 정리했습니다.")


if __name__ == "__main__":
    main()