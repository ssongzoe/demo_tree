#!/usr/bin/env python3
"""RB-Y1 tote 인식/파지, 모바일 이송, AR 정렬, 배치 및 복귀 통합 데모.

동작 순서
0. torso / head를 calibration 기준 자세로 먼저 맞춘 뒤 Tote D435와 AR RealSense를 한 번만 시작하고 계속 streaming
1. 현재 자세에서 D435 영상의 TOP + TL feature로 tote one-shot 정렬
2. GRASP 자세로 진입한 뒤 그리퍼를 닫고 UP 자세로 들어 올림
3. BACK_TARGET → TURN_TARGET → STRAIGHT_TARGET 순서로 기존 이송 경로 수행
4. AR 마커 기준으로 배치 위치 정렬
5. UP → GRASP로 내려놓고 그리퍼를 연 뒤 BEFORE 자세로 후퇴
6. RETURN_BACK_TARGET → RETURN_TURN_TARGET → RETURN_STRAIGHT_TARGET 순서로 기존 복귀 경로 수행
7. Tote / AR 카메라는 시작할 때 한 번만 켜고 계속 유지하며, 복귀가 끝나면 같은 과정을 즉시 다시 시작

이동 거리와 회전각은 아래 target 상수만 수정하면 되며, 실행 로그는 target 값을 직접 읽어 출력하므로 값과 설명이 따로 어긋나지 않는다.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from control.gripper_controller import GripperController
from control.mobile_controller import OdometryMonitor, build_leg, initialize_mobile, move_leg, odom_pose, wait_for_odometry
from control.robot_controller import move_both_arms, move_torso_and_head
from skills.ar_align import ARAligner
from skills.tote_align import ToteAligner


# -----------------------------------------------------------------------------
# 로봇 / 카메라 설정
# -----------------------------------------------------------------------------

ADDRESS = "192.168.30.1:50051"

TOTE_CAMERA_SERIAL = "250122079439"
AR_CAMERA_SERIAL = "409122274689"
MARKER_ID = 8

DEFAULT_GRIPPER_TARGET = 0.80
DEFAULT_GRIPPER_TORQUE = 0.20

# Tote vision과 grasp pose는 이 torso / head 기준으로 맞춰져 있으므로 프로그램 시작 시 한 번 정확히 고정한다.
INITIAL_TORSO = np.deg2rad([0.0, 30.0, -50.0, 30.0, 0.0, 0.0]).tolist()
INITIAL_HEAD = np.deg2rad([0.0, 43.0]).tolist()


# -----------------------------------------------------------------------------
# Tote 파지 자세
# torso는 BEFORE / GRASP / UP 동안 동일하게 유지하며, 아래 값 하나만 사용한다.
# -----------------------------------------------------------------------------

BEFORE_RIGHT = np.deg2rad([-51.651, -35.387, -16.519, -42.941, -31.167, 73.404, 0.001]).tolist()
BEFORE_LEFT = np.deg2rad([-51.625, 37.742, 19.947, -44.127, 35.084, 75.497, -0.033]).tolist()

GRASP_RIGHT = np.deg2rad([-53.243, -27.593, -16.509, -45.481, -31.781, 73.370, 0.012]).tolist()
GRASP_LEFT = np.deg2rad([-51.643, 29.044, 19.947, -45.832, 35.513, 75.498, -0.036]).tolist()

UP_RIGHT = np.deg2rad([-20.43, -25.28, -27.43, -98.12, -52.06, 91.75, -15.12]).tolist()
UP_LEFT = np.deg2rad([-20.43, 25.28, 27.43, -98.12, 52.06, 91.75, 15.12]).tolist()



# -----------------------------------------------------------------------------
# 모바일 경로
# target = (x [m], y [m], yaw [rad])
# 아래 target 값만 수정하면 실제 실행 로그도 현재 값에 맞춰 자동으로 바뀐다.
# -----------------------------------------------------------------------------

BACK_TARGET = (-0.10, 0.0, 0.0)
TURN_TARGET = (-0.05, -0.05, math.radians(-180.43))
STRAIGHT_TARGET = (0.65, 0.0, 0.0)

RETURN_BACK_TARGET = (-0.35, 0.0, 0.0)
RETURN_TURN_TARGET = (0.0, 0.0, math.radians(179.43))
RETURN_STRAIGHT_TARGET = (0.82, 0.0, 0.0)


def describe_target(target) -> str:
    """모바일 target의 현재 x/y/yaw 값을 실행 로그에 사용할 문자열로 변환한다."""
    x_m, y_m, yaw_rad = target
    return f"x={x_m:+.2f} m, y={y_m:+.2f} m, yaw={math.degrees(yaw_rad):+.2f} deg"


def run_mobile_leg(robot, monitor, stream, step: str, target, duration: float, stop_at_end: bool, settle: float) -> bool:
    """현재 odometry 자세를 기준으로 상대 이동을 한 번 수행하며, 설정된 target 값을 그대로 로그에 표시한다."""
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
    stream = robot.create_command_stream(priority=10)

    try:
        if not run_mobile_leg(robot, monitor, stream, "이송 1/3", BACK_TARGET, 2.0, False, 0.0):
            return False

        if not run_mobile_leg(robot, monitor, stream, "이송 2/3", TURN_TARGET, 7.0, False, 0.0):
            return False

        if not run_mobile_leg(robot, monitor, stream, "이송 3/3", STRAIGHT_TARGET, 5.0, True, 0.2):
            return False

        return True

    finally:
        stream.cancel()
        stream.wait_for(500)


def run_return_route(robot, monitor) -> bool:
    """박스를 놓고 팔을 BEFORE로 후퇴한 뒤 RETURN_BACK → RETURN_TURN → RETURN_STRAIGHT target을 기존 설정 그대로 수행한다."""
    stream = robot.create_command_stream(priority=10)

    try:
        legs = [
            ("복귀 1/3", RETURN_BACK_TARGET, 3.0, False, 0.0),
            ("복귀 2/3", RETURN_TURN_TARGET, 10.0, False, 0.0),
            ("복귀 3/3", RETURN_STRAIGHT_TARGET, 5.0, True, 0.2),
        ]

        for step, target, duration, stop_at_end, settle in legs:
            if not run_mobile_leg(robot, monitor, stream, step, target, duration, stop_at_end, settle):
                print(f"{step} 실패: {describe_target(target)}")
                return False

        return True

    finally:
        stream.cancel()
        stream.wait_for(500)


def detect_grasp_and_lift(
    robot,
    monitor,
    gripper,
    tote_aligner: ToteAligner,
    gripper_target: float,
    gripper_torque: float,
) -> bool:
    """현재 자세에서 tote를 인식해 base를 정렬한 뒤 GRASP로 진입하고, 그리퍼를 닫아 UP 자세로 들어 올린다."""
    print("[1/3] 현재 자세에서 Tote 영상 인식 + one-shot 정렬")
    if not tote_aligner.align(robot, monitor, verify=True):
        print("Tote one-shot 정렬 실패")
        return False

    print("[2/3] 현재 자세 → GRASP")
    if not move_both_arms(robot, GRASP_RIGHT, GRASP_LEFT, minimum_time=1.0):
        print("GRASP 자세 이동 실패")
        return False

    print(f"그리퍼 닫기: target={gripper_target:.2f}, torque={gripper_torque:.2f} Nm")
    gripper.close(target=gripper_target, torque=gripper_torque, duration=1.0)
    print(f"그리퍼 현재 위치: {gripper.get_positions().round(3)}")

    print("[3/3] GRASP → UP")
    if not move_both_arms(robot, UP_RIGHT, UP_LEFT, minimum_time=1.0):
        print("UP 자세 이동 실패")
        return False

    return True


def lower_release_and_retract(robot, gripper) -> bool:
    """AR 정렬 후 UP에서 GRASP로 내려놓고 그리퍼를 연 뒤, 손잡이에서 빠져나오도록 BEFORE 자세로 양팔을 후퇴한다."""
    print("[5/6] UP → GRASP")
    if not move_both_arms(robot, GRASP_RIGHT, GRASP_LEFT, minimum_time=1.0):
        print("GRASP 자세 이동 실패")
        return False

    print("그리퍼 열기")
    gripper.open(duration=1.0)

    print("GRASP → BEFORE")
    if not move_both_arms(robot, BEFORE_RIGHT, BEFORE_LEFT, minimum_time=2.0):
        print("BEFORE 자세 이동 실패")
        return False

    return True



def run_cycle(robot, monitor, gripper, tote_aligner, ar_aligner, args, cycle_index: int) -> None:
    """박스 인식/파지부터 이송, AR 정렬, 배치, 복귀까지 한 사이클을 수행하며 완료 후 다음 사이클을 같은 위치에서 시작한다."""
    print(f"\n{'=' * 24} CYCLE {cycle_index} START {'=' * 24}")
    print(f"박스 파지 시작")
    if not detect_grasp_and_lift(
        robot,
        monitor,
        gripper,
        tote_aligner=tote_aligner,
        gripper_target=args.gripper_target,
        gripper_torque=args.gripper_torque,
    ):
        raise RuntimeError("Tote 인식 / 정렬 / 파지 실패")

    print(f"이송 시작: {describe_target(BACK_TARGET)} → {describe_target(TURN_TARGET)} → {describe_target(STRAIGHT_TARGET)}")
    if not run_turn_and_go(robot, monitor):
        raise RuntimeError("Turn-and-go 실패")

    print("AR 마커 one-shot 정렬")
    if not ar_aligner.align(robot, monitor):
        raise RuntimeError("AR 마커 정렬 실패")

    if not lower_release_and_retract(robot, gripper):
        raise RuntimeError("Tote 배치 실패")

    print(
        f"복귀 시작: {describe_target(RETURN_BACK_TARGET)} → {describe_target(RETURN_TURN_TARGET)} → "
        f"{describe_target(RETURN_STRAIGHT_TARGET)}"
    )
    if not run_return_route(robot, monitor):
        raise RuntimeError("복귀 주행 실패")

    print(f"{'=' * 24} CYCLE {cycle_index} DONE {'=' * 25}")
    time.sleep(1.0)  # 다음 사이클 시작 전 잠시 대기

def main() -> None:
    parser = argparse.ArgumentParser(description="RB-Y1 tote vision full sequence 반복 데모")
    parser.add_argument("--address", default=ADDRESS, help="로봇 주소")
    parser.add_argument("--model", choices=("a", "m"), default="m", help="RB-Y1 모델")
    parser.add_argument("--marker-id", type=int, default=MARKER_ID, help="배치 위치 정렬에 사용할 AR 마커 ID")
    parser.add_argument("--tote-camera-serial", default=TOTE_CAMERA_SERIAL, help="Tote 인식/정렬용 D435 serial")
    parser.add_argument("--ar-camera-serial", default=AR_CAMERA_SERIAL, help="AR 마커 정렬용 RealSense serial")
    parser.add_argument("--show-tote", action="store_true", help="Tote 검출 OpenCV 화면 표시")
    parser.add_argument("--gripper-target", type=float, default=DEFAULT_GRIPPER_TARGET, help="그리퍼 닫힘 위치")
    parser.add_argument("--gripper-torque", type=float, default=DEFAULT_GRIPPER_TORQUE, help="그리퍼 파지 토크 [Nm]")
    parser.add_argument("--cycles", type=int, default=0, help="반복 횟수, 0이면 Ctrl+C 전까지 무한 반복")
    args = parser.parse_args()

    robot = initialize_mobile(args.address, args.model, power=".*", servo=".*", unlimited=False)

    gripper = None
    tote_aligner = ToteAligner(camera_serial=args.tote_camera_serial, show=args.show_tote)
    ar_aligner = ARAligner(marker_id=args.marker_id, camera_serial=args.ar_camera_serial)
    monitor = OdometryMonitor()
    state_update_started = False
    completed_cycles = 0

    try:
        robot.set_tool_flange_output_voltage("right", 12)
        robot.set_tool_flange_output_voltage("left", 12)
        time.sleep(0.5)

        gripper = GripperController(position_torque=args.gripper_torque)
        gripper.connect()
        gripper.open(duration=2.0)

        robot.start_state_update(monitor.on_state, rate=50)
        state_update_started = True

        if not wait_for_odometry(monitor):
            raise RuntimeError("Odometry를 받지 못했습니다.")

        # Tote 검출과 파지 자세가 torso/head 기준에 민감하므로 카메라 측정 전에 기준 자세를 한 번 확실하게 맞춘다.
        print("초기 Torso / Head 기준 자세로 이동")
        if not move_torso_and_head(robot, INITIAL_TORSO, INITIAL_HEAD, minimum_time=2.0):
            raise RuntimeError("초기 Torso / Head 자세 이동 실패")

        # 두 RealSense는 프로그램 시작 시 한 번만 켜고 모든 cycle에서 stream을 계속 유지한다.
        print("Tote / AR 카메라 시작")
        tote_aligner.start()
        ar_aligner.start()

        cycle_index = 1

        while args.cycles == 0 or cycle_index <= args.cycles:
            run_cycle(robot, monitor, gripper, tote_aligner, ar_aligner, args, cycle_index)
            completed_cycles += 1
            cycle_index += 1

        print(f"요청한 {completed_cycles}개 사이클 완료")

    except KeyboardInterrupt:
        print(f"\n사용자가 반복 데모를 중단했습니다. 완료 사이클: {completed_cycles}")

    except Exception as error:
        print(f"반복 데모 실패: {error} | 완료 사이클: {completed_cycles}")

    finally:
        tote_aligner.stop()
        ar_aligner.stop()

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
        print("모든 연결과 제어를 정리했습니다.")


if __name__ == "__main__":
    main()