#!/usr/bin/env python3
"""Tote one-shot 정렬 후 파지하고 들어 올리는 데모.

동작 순서
1. 로봇 / 그리퍼 준비
2. ALIGN_GRASP 자세로 이동
3. TOP + TL 기반 one-shot base 정렬
4. 그리퍼 닫기
5. 양팔을 LIFT 자세로 이동

※ torso는 ALIGN / GRASP / LIFT 동안 동일한 자세를 유지한다.
※ demo끼리 import하지 않고 control / skills만 사용한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.gripper_controller import GripperController
from control.mobile_controller import OdometryMonitor, initialize_mobile, wait_for_odometry
from control.robot_controller import READY_POSE, move_both_arms, move_to_upper_body_pose
from skills.tote_align import align_tote


ADDRESS = "192.168.30.1:50051"
CAMERA_SERIAL = "250122079439"

DEFAULT_GRIPPER_TARGET = 0.35
DEFAULT_GRIPPER_TORQUE = 0.20


# ALIGN / GRASP 동안 유지할 상체 자세
TORSO_POSE = np.deg2rad([
    0.0, 30.0, -50.0, 30.0, 0.0, 0.0,
]).tolist()

ALIGN_GRASP_RIGHT = np.deg2rad([
    -53.243, -27.593, -16.509, -45.481, -31.781, 73.370, 0.012,
]).tolist()

ALIGN_GRASP_LEFT = np.deg2rad([
    -51.643, 29.044, 19.947, -45.832, 35.513, 75.498, -0.036,
]).tolist()


# 파지 후 들어 올릴 양팔 자세
LIFT_RIGHT = np.deg2rad([
    -56.214, -32.637, -20.526, -39.769, -47.012, 73.779, 0.001,
]).tolist()

LIFT_LEFT = np.deg2rad([
    -56.215, 32.642, 20.524, -39.767, 47.012, 73.780, 0.0,
]).tolist()


ALIGN_GRASP_POSE = {
    "torso": TORSO_POSE,
    "right_arm": ALIGN_GRASP_RIGHT,
    "left_arm": ALIGN_GRASP_LEFT,
    "head": READY_POSE["head"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="RB-Y1 tote one-shot pickup demo")
    parser.add_argument("--address", default=ADDRESS, help="로봇 주소")
    parser.add_argument("--model", choices=("a", "m"), default="m", help="RB-Y1 모델")
    parser.add_argument("--camera-serial", default=CAMERA_SERIAL, help="Tote 정렬용 D435 serial")
    parser.add_argument("--show", action="store_true", help="OpenCV 검출 화면 표시")
    parser.add_argument("--gripper-target", type=float, default=DEFAULT_GRIPPER_TARGET, help="그리퍼 닫힘 위치")
    parser.add_argument("--gripper-torque", type=float, default=DEFAULT_GRIPPER_TORQUE, help="그리퍼 파지 토크 [Nm]")
    args = parser.parse_args()

    robot = initialize_mobile(
        args.address,
        args.model,
        power=".*",
        servo=".*",
        unlimited=False,
    )

    gripper = None
    monitor = OdometryMonitor()
    state_update_started = False

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

        print("[1/4] ALIGN / GRASP 자세로 이동")
        if not move_to_upper_body_pose(robot, ALIGN_GRASP_POSE, minimum_time=2.0):
            raise RuntimeError("ALIGN / GRASP 자세 이동 실패")

        print("[2/4] Tote one-shot 정렬")
        if not align_tote(robot, monitor, camera_serial=args.camera_serial, verify=True, show=args.show):
            raise RuntimeError("Tote one-shot 정렬 실패")

        print(f"[3/4] 그리퍼 닫기: target={args.gripper_target:.2f}, torque={args.gripper_torque:.2f} Nm")
        gripper.close(target=args.gripper_target, torque=args.gripper_torque, duration=1.0)
        print(f"그리퍼 현재 위치: {gripper.get_positions().round(3)}")

        print("[4/4] Tote 들어 올리기")
        if not move_both_arms(robot, LIFT_RIGHT, LIFT_LEFT, minimum_time=2.0):
            raise RuntimeError("LIFT 자세 이동 실패")

        print("Tote pickup 완료")

    except KeyboardInterrupt:
        print("\n사용자가 데모를 중단했습니다.")

    except Exception as error:
        print(f"데모 실패: {error}")

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
        print("로봇과 그리퍼 연결 종료")


if __name__ == "__main__":
    main()
