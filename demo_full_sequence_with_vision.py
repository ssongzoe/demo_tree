#!/usr/bin/env python3
"""RB-Y1 tote 인식/파지, 모바일 이송, AR 정렬, 배치 및 복귀 통합 데모.

동작 순서
1. ALIGN_GRASP 상체 자세로 이동
2. D435 영상의 TOP + TL feature로 tote one-shot 정렬
3. 그리퍼를 닫고 LIFT 자세로 들어 올림
4. BACK_TARGET → TURN_TARGET → STRAIGHT_TARGET 순서로 이송
5. AR 마커 기준 정렬 후 LIFT → ALIGN_GRASP로 내려놓고 그리퍼를 연 뒤 BEFORE_GRASP로 후퇴
6. RETURN_BACK_TARGET → RETURN_TURN_TARGET → RETURN_STRAIGHT_TARGET 순서로 복귀

이동 거리와 회전각은 아래 target 상수만 수정하면 되며, 실행 로그는 target 값을 직접 읽어 출력하므로 값과 설명이 따로 어긋나지 않는다.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from control.gripper_controller import GripperController
from control.mobile_controller import OdometryMonitor, build_leg, initialize_mobile, move_leg, odom_pose, wait_for_odometry
from control.robot_controller import READY_POSE, move_both_arms, move_to_upper_body_pose
from skills.ar_align import align_to_marker
from skills.tote_align import align_tote


# -----------------------------------------------------------------------------
# 로봇 / 카메라 설정
# -----------------------------------------------------------------------------

ADDRESS = "192.168.30.1:50051"

TOTE_CAMERA_SERIAL = "250122079439"
AR_CAMERA_SERIAL = "409122274689"
MARKER_ID = 8

DEFAULT_GRIPPER_TARGET = 0.80
DEFAULT_GRIPPER_TORQUE = 0.20


# -----------------------------------------------------------------------------
# Tote 인식 / 파지 자세
# -----------------------------------------------------------------------------

# 카메라로 tote를 보고 정렬한 뒤 그대로 파지할 때 유지하는 torso 자세
TORSO_POSE = np.deg2rad([0.0, 30.0, -50.0, 30.0, 0.0, 0.0]).tolist()

ALIGN_GRASP_RIGHT = np.deg2rad([-53.243, -27.593, -16.509, -45.481, -31.781, 73.370, 0.012]).tolist()
ALIGN_GRASP_LEFT = np.deg2rad([-51.643, 29.044, 19.947, -45.832, 35.513, 75.498, -0.036]).tolist()

# 파지 후 들어 올릴 때는 torso를 유지하고 양팔만 LIFT 자세로 이동한다.
LIFT_RIGHT = np.deg2rad([-56.214, -32.637, -20.526, -39.769, -47.012, 73.779, 0.001]).tolist()
LIFT_LEFT = np.deg2rad([-56.215, 32.642, 20.524, -39.767, 47.012, 73.780, 0.0]).tolist()

# 배치 후 열린 그리퍼를 tote 손잡이에서 빼낼 때 사용할 후퇴 자세
BEFORE_GRASP_RIGHT = np.deg2rad([-31.0, -54.0, -10.0, -91.0, -57.0, 87.0, -13.0]).tolist()
BEFORE_GRASP_LEFT = np.deg2rad([-31.0, 54.0, 10.0, -91.0, 57.0, 87.0, 13.0]).tolist()

ALIGN_GRASP_POSE = {
    "torso": TORSO_POSE,
    "right_arm": ALIGN_GRASP_RIGHT,
    "left_arm": ALIGN_GRASP_LEFT,
    "head": READY_POSE["head"],
}


# -----------------------------------------------------------------------------
# 배치 동작
# -----------------------------------------------------------------------------

# Tote를 내려놓을 때는 LIFT → ALIGN_GRASP까지 내려간 뒤 그리퍼를 열고, 손잡이에서 빠져나오기 위해 BEFORE_GRASP로 후퇴한다.


# -----------------------------------------------------------------------------
# 모바일 경로
# target = (x [m], y [m], yaw [rad])
# 아래 값만 바꾸면 실제 실행 로그도 자동으로 새 값에 맞춰 출력된다.
# -----------------------------------------------------------------------------

BACK_TARGET = (-0.10, 0.0, 0.0)
TURN_TARGET = (-0.50, -0.05, math.radians(-185.43))
STRAIGHT_TARGET = (0.75, 0.0, 0.0)

RETURN_BACK_TARGET = (-0.35, 0.0, 0.0)
RETURN_TURN_TARGET = (0.0, 0.0, math.radians(179.43))
RETURN_STRAIGHT_TARGET = (1.15, -0.20, 0.0)


def describe_target(target) -> str:
    """모바일 target의 현재 x/y/yaw 값을 사람이 읽기 쉬운 문자열로 만든다."""
    x_m, y_m, yaw_rad = target
    return f"x={x_m:+.2f} m, y={y_m:+.2f} m, yaw={math.degrees(yaw_rad):+.2f} deg"


def run_mobile_leg(robot, monitor, stream, step: str, target, duration: float, stop_at_end: bool, settle: float) -> bool:
    """현재 odometry 자세를 기준으로 설정된 target까지 상대 이동하고, 실제 target 값을 로그에 함께 출력한다."""
    print(f"{step}: {describe_target(target)}")

    leg = build_leg(
        start=odom_pose(monitor.odom),
        target=target,
        absolute=False,
        duration=duration,
        turn_direction="shortest",
    )

    return move_leg(robot, monitor, leg, settle=settle, stream=stream, stop_at_end=stop_at_end)


def run_turn_and_go(robot, monitor) -> bool:
    """출발 지점에서 BACK → TURN → STRAIGHT target을 하나의 command stream으로 연속 수행한다."""
    stream = robot.create_command_stream(priority=10)

    try:
        legs = [
            ("이송 1/3", BACK_TARGET, 2.0, False, 0.0),
            ("이송 2/3", TURN_TARGET, 7.0, False, 0.0),
            ("이송 3/3", STRAIGHT_TARGET, 5.0, True, 1.5),
        ]

        for step, target, duration, stop_at_end, settle in legs:
            if not run_mobile_leg(robot, monitor, stream, step, target, duration, stop_at_end, settle):
                print(f"{step} 실패: {describe_target(target)}")
                return False

        return True

    finally:
        stream.cancel()
        stream.wait_for(500)


def run_return_route(robot, monitor) -> bool:
    """배치 후 RETURN_BACK → RETURN_TURN → RETURN_STRAIGHT target을 하나의 command stream으로 연속 수행한다."""
    stream = robot.create_command_stream(priority=10)

    try:
        legs = [
            ("복귀 1/3", RETURN_BACK_TARGET, 3.0, False, 0.0),
            ("복귀 2/3", RETURN_TURN_TARGET, 10.0, False, 0.0),
            ("복귀 3/3", RETURN_STRAIGHT_TARGET, 5.0, True, 1.5),
        ]

        for step, target, duration, stop_at_end, settle in legs:
            if not run_mobile_leg(robot, monitor, stream, step, target, duration, stop_at_end, settle):
                print(f"{step} 실패: {describe_target(target)}")
                return False

        return True

    finally:
        stream.cancel()
        stream.wait_for(500)


def align_grasp_and_lift(robot, monitor, gripper, tote_camera_serial: str, gripper_target: float, gripper_torque: float, show: bool) -> bool:
    """정렬/파지 상체 자세에서 tote를 인식해 base를 one-shot 보정하고, 그리퍼를 닫은 뒤 양팔을 LIFT 자세로 이동한다."""
    print("[1/7] ALIGN / GRASP 상체 자세로 이동")
    if not move_to_upper_body_pose(robot, ALIGN_GRASP_POSE, minimum_time=2.0):
        print("ALIGN / GRASP 상체 자세 이동 실패")
        return False

    print("[2/7] Tote 영상 인식 + one-shot 정렬")
    if not align_tote(robot, monitor, camera_serial=tote_camera_serial, verify=True, show=show):
        print("Tote one-shot 정렬 실패")
        return False

    print(f"[3/7] 그리퍼 닫기: target={gripper_target:.2f}, torque={gripper_torque:.2f} Nm")
    gripper.close(target=gripper_target, torque=gripper_torque, duration=1.0)
    print(f"그리퍼 현재 위치: {gripper.get_positions().round(3)}")

    print("[4/7] Tote 들어 올리기")
    if not move_both_arms(robot, LIFT_RIGHT, LIFT_LEFT, minimum_time=2.0):
        print("LIFT 자세 이동 실패")
        return False

    return True


def lower_release_and_retract(robot, gripper) -> bool:
    """LIFT에서 ALIGN_GRASP까지 tote를 내리고 그리퍼를 연 뒤, 손잡이에 남은 그리퍼를 BEFORE_GRASP 자세로 빼낸다."""
    print("[6/7] Tote 내려놓기: LIFT → ALIGN_GRASP")
    if not move_both_arms(robot, ALIGN_GRASP_RIGHT, ALIGN_GRASP_LEFT, minimum_time=2.0):
        print("ALIGN_GRASP 자세 이동 실패")
        return False

    print("그리퍼 열기")
    gripper.open(duration=1.0)

    print("손잡이에서 양팔 후퇴: ALIGN_GRASP → BEFORE_GRASP")
    if not move_both_arms(robot, BEFORE_GRASP_RIGHT, BEFORE_GRASP_LEFT, minimum_time=2.0):
        print("BEFORE_GRASP 후퇴 자세 이동 실패")
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="RB-Y1 tote 인식 기반 full sequence 데모")
    parser.add_argument("--address", default=ADDRESS, help="로봇 주소")
    parser.add_argument("--model", choices=("a", "m"), default="m", help="RB-Y1 모델")
    parser.add_argument("--marker-id", type=int, default=MARKER_ID, help="배치 위치 정렬에 사용할 AR 마커 ID")
    parser.add_argument("--tote-camera-serial", default=TOTE_CAMERA_SERIAL, help="Tote 인식/정렬용 D435 serial")
    parser.add_argument("--ar-camera-serial", default=AR_CAMERA_SERIAL, help="AR 마커 정렬용 RealSense serial")
    parser.add_argument("--show-tote", action="store_true", help="Tote 검출 OpenCV 화면 표시")
    parser.add_argument("--gripper-target", type=float, default=DEFAULT_GRIPPER_TARGET, help="그리퍼 닫힘 위치, 0.0=열림, 1.0=완전 닫힘")
    parser.add_argument("--gripper-torque", type=float, default=DEFAULT_GRIPPER_TORQUE, help="그리퍼 파지 토크 [Nm], 최대 0.46")
    args = parser.parse_args()

    # Full demo는 모바일과 양팔을 모두 사용하므로 전체 servo를 준비한다.
    robot = initialize_mobile(args.address, args.model, power=".*", servo=".*", unlimited=False)

    gripper = None
    monitor = OdometryMonitor()
    state_update_started = False
    demo_completed = False

    try:
        robot.set_tool_flange_output_voltage("right", 12)
        robot.set_tool_flange_output_voltage("left", 12)
        time.sleep(0.5)

        gripper = GripperController(position_torque=args.gripper_torque)
        gripper.connect()
        gripper.open(duration=2.0)

        # Tote one-shot 정렬부터 mobile_controller의 odometry가 필요하므로 파지보다 먼저 state update를 시작한다.
        robot.start_state_update(monitor.on_state, rate=50)
        state_update_started = True

        if not wait_for_odometry(monitor):
            raise RuntimeError("Odometry를 받지 못했습니다.")

        if not align_grasp_and_lift(
            robot,
            monitor,
            gripper,
            tote_camera_serial=args.tote_camera_serial,
            gripper_target=args.gripper_target,
            gripper_torque=args.gripper_torque,
            show=args.show_tote,
        ):
            raise RuntimeError("Tote 인식 / 정렬 / 파지 실패")

        print(f"[5/7] Tote 이송 시작: {describe_target(BACK_TARGET)} → {describe_target(TURN_TARGET)} → {describe_target(STRAIGHT_TARGET)}")
        if not run_turn_and_go(robot, monitor):
            raise RuntimeError("Turn-and-go 실패")

        print(f"AR 마커 정렬: marker_id={args.marker_id}, camera={args.ar_camera_serial}")
        if not align_to_marker(robot, monitor, marker_id=args.marker_id, camera_serial=args.ar_camera_serial):
            raise RuntimeError("AR 마커 정렬 실패")

        if not lower_release_and_retract(robot, gripper):
            raise RuntimeError("Tote 배치 실패")

        print(
            f"[7/7] 복귀 시작: {describe_target(RETURN_BACK_TARGET)} → {describe_target(RETURN_TURN_TARGET)} → "
            f"{describe_target(RETURN_STRAIGHT_TARGET)}"
        )
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

        try:
            robot.set_tool_flange_output_voltage("right", 0)
            robot.set_tool_flange_output_voltage("left", 0)
            robot.disable_control_manager()
        except Exception:
            pass

        robot.disconnect()

        if demo_completed:
            print("모든 연결과 제어를 정상 종료했습니다.")
        else:
            print("오류 종료 후 연결과 제어를 정리했습니다.")


if __name__ == "__main__":
    main()