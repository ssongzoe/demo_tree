#!/usr/bin/env python3
"""RB-Y1 tote 인식/파지, 모바일 이송, AR 정렬, 배치 및 복귀 통합 데모.

동작 순서
1. BEFORE 자세로 이동하고 D435 영상의 TOP + TL feature로 tote one-shot 정렬
2. GRASP 자세로 진입한 뒤 그리퍼를 닫고 UP 자세로 들어 올림
3. BACK_TARGET → TURN_TARGET → STRAIGHT_TARGET 순서로 기존 이송 경로 수행
4. AR 마커 기준으로 배치 위치 정렬
5. UP → GRASP로 내려놓고 그리퍼를 연 뒤 BEFORE 자세로 후퇴
6. RETURN_BACK_TARGET → RETURN_TURN_TARGET → RETURN_STRAIGHT_TARGET 순서로 기존 복귀 경로 수행

이동 거리와 회전각은 아래 target 상수만 수정하면 되며, 실행 로그는 target 값을 직접 읽어 출력하므로 값과 설명이 따로 어긋나지 않는다.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from control.gripper_controller import GripperController
from control.mobile_controller import OdometryMonitor, build_leg, initialize_mobile, move_leg, odom_pose, wait_for_odometry
from control.robot_controller import READY_POSE, move_both_arms
from skills.ar_align import ARAligner
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
# Tote 파지 자세
# torso는 BEFORE / GRASP / UP 동안 동일하게 유지하며, 아래 값 하나만 사용한다.
# -----------------------------------------------------------------------------

# TORSO_POSE = np.deg2rad([0.0, 30.0, -50.0, 30.0, 0.0, 0.0]).tolist()

BEFORE_RIGHT = np.deg2rad([-51.651, -35.387, -16.519, -42.941, -31.167, 73.404, 0.001]).tolist()
BEFORE_LEFT = np.deg2rad([-51.625, 37.742, 19.947, -44.127, 35.084, 75.497, -0.033]).tolist()

GRASP_RIGHT = np.deg2rad([-53.243, -27.593, -16.509, -45.481, -31.781, 73.370, 0.012]).tolist()
GRASP_LEFT = np.deg2rad([-51.643, 29.044, 19.947, -45.832, 35.513, 75.498, -0.036]).tolist()

UP_RIGHT = np.deg2rad([-56.214, -32.637, -20.526, -39.769, -47.012, 73.779, 0.001]).tolist()
UP_LEFT = np.deg2rad([-56.215, 32.642, 20.524, -39.767, 47.012, 73.780, 0.0]).tolist()



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


def run_turn_and_go(robot, monitor, ar_aligner: ARAligner) -> bool:
    stream = robot.create_command_stream(priority=10)

    try:
        if not run_mobile_leg(robot, monitor, stream, "이송 1/3", BACK_TARGET, 2.0, False, 0.0):
            return False

        if not run_mobile_leg(robot, monitor, stream, "이송 2/3", TURN_TARGET, 7.0, False, 0.0):
            return False

        ar_aligner.start()

        if not run_mobile_leg(robot, monitor, stream, "이송 3/3", STRAIGHT_TARGET, 5.0, True, 0.5):
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
    tote_camera_serial: str,
    gripper_target: float,
    gripper_torque: float,
    show: bool,
) -> bool:
    """BEFORE 자세에서 tote를 인식해 base를 정렬한 뒤 GRASP로 진입하고, 그리퍼를 닫아 UP 자세로 들어 올린다."""
    print("[1/3] 현재 자세에서 Tote 영상 인식 + one-shot 정렬")
    if not align_tote(robot, monitor, camera_serial=tote_camera_serial, verify=True, show=show):
        print("Tote one-shot 정렬 실패")
        return False

    print("[2/3] BEFORE → GRASP")
    if not move_both_arms(robot, GRASP_RIGHT, GRASP_LEFT, minimum_time=0.5):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="RB-Y1 tote 인식 기반 full sequence 데모")
    parser.add_argument("--address", default=ADDRESS, help="로봇 주소")
    parser.add_argument("--model", choices=("a", "m"), default="m", help="RB-Y1 모델")
    parser.add_argument("--marker-id", type=int, default=MARKER_ID, help="배치 위치 정렬에 사용할 AR 마커 ID")
    parser.add_argument("--tote-camera-serial", default=TOTE_CAMERA_SERIAL, help="Tote 인식/정렬용 D435 serial")
    parser.add_argument("--ar-camera-serial", default=AR_CAMERA_SERIAL, help="AR 마커 정렬용 RealSense serial")
    parser.add_argument("--show-tote", action="store_true", help="Tote 검출 OpenCV 화면 표시")
    parser.add_argument("--gripper-target", type=float, default=DEFAULT_GRIPPER_TARGET, help="그리퍼 닫힘 위치")
    parser.add_argument("--gripper-torque", type=float, default=DEFAULT_GRIPPER_TORQUE, help="그리퍼 파지 토크 [Nm]")
    args = parser.parse_args()

    # Full demo는 모바일과 양팔을 모두 사용하므로 wheel만이 아니라 전체 servo를 준비한다.
    robot = initialize_mobile(args.address, args.model, power=".*", servo=".*", unlimited=False)

    gripper = None
    ar_aligner = ARAligner(marker_id=args.marker_id, camera_serial=args.ar_camera_serial)
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

        # 첫 단계의 tote one-shot 정렬부터 odometry가 필요하므로 BEFORE 자세 이동과 영상 인식보다 먼저 state update를 시작한다.
        robot.start_state_update(monitor.on_state, rate=50)
        state_update_started = True

        if not wait_for_odometry(monitor):
            raise RuntimeError("Odometry를 받지 못했습니다.")
        # 박스 잡기 
        if not detect_grasp_and_lift(
            robot,
            monitor,
            gripper,
            tote_camera_serial=args.tote_camera_serial,
            gripper_target=args.gripper_target,
            gripper_torque=args.gripper_torque,
            show=args.show_tote,
        ):
            raise RuntimeError("Tote 인식 / 정렬 / 파지 실패")

        print(f"이송 시작: {describe_target(BACK_TARGET)} → {describe_target(TURN_TARGET)} → {describe_target(STRAIGHT_TARGET)}")
        if not run_turn_and_go(robot, monitor, ar_aligner):
            raise RuntimeError("Turn-and-go 실패")

        print("AR 마커 one-shot 정렬")
        if not ar_aligner.align(robot, monitor):
            raise RuntimeError("AR 마커 정렬 실패")

        if not lower_release_and_retract(robot, gripper):
            raise RuntimeError("Tote 배치 실패")

        print(
            f"[6/6] 복귀 시작: {describe_target(RETURN_BACK_TARGET)} → {describe_target(RETURN_TURN_TARGET)} → "
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

        if demo_completed:
            print("모든 연결과 제어를 정상 종료했습니다.")
        else:
            print("오류 종료 후 연결과 제어를 정리했습니다.")


if __name__ == "__main__":
    main()