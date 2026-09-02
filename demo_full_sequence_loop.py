"""RB-Y1 tote 인식/파지, 모바일 이송, AR 정렬, 배치 및 복귀 통합 데모.

동작 순서
0. torso / head를 calibration 기준 자세로 먼저 맞춘 뒤, main에서 Head RealSense pipeline을 한 번만 시작해 Tote/AR가 함께 사용
1. 현재 자세에서 D435 영상의 TOP + TL feature로 tote one-shot 정렬
2. Tote 정렬 후 양팔을 BEFORE → GRASP로 이동한 뒤 그리퍼를 닫고 UP 자세로 들어 올림
3. BACK + TURN + STRAIGHT를 합성한 direct target으로 이송하며 후반에 Head를 정면 자세로 전환
4. AR 마커 기준으로 배치 위치 정렬
5. UP → GRASP로 내려놓고 그리퍼를 연 뒤 BEFORE 자세로 후퇴
6. 복귀는 BACK을 단독 수행한 뒤 TURN + STRAIGHT를 합성한 direct target으로 이동
7. Head RealSense pipeline의 start / stop 소유권은 main에만 두고, Tote / AR aligner는 전달받은 공용 stream에서 frame만 읽음
8. 로봇 상태는 기존 SDK callback에서 WCS publisher에도 전달하고, 작업 상태와 함께 별도 스레드에서 주기 전송

이동 거리와 회전각은 아래 target 상수만 수정하면 되며, 실행 로그는 target 값을 직접 읽어 출력하므로 값과 설명이 따로 어긋나지 않는다.
"""

import argparse
import math
import threading
import time
from concurrent.futures import Future

import numpy as np

from communication.wcs.publisher import WcsPublisher
from control.gripper_controller import GripperController
from control.mobile_controller import OdometryMonitor, build_leg, initialize_mobile, move_leg, odom_pose, wait_for_odometry
from control.robot_controller import move_both_arms, move_torso_and_head
from skills.ar_align import ARAligner
from skills.tote_align import ToteAligner
from utils.ar_marker import RealSenseCamera

# -----------------------------------------------------------------------------
# 로봇 / 카메라 설정
# -----------------------------------------------------------------------------

ADDRESS = "192.168.30.1:50051"

HEAD_CAMERA_SERIAL = "250122079439"
MARKER_ID = 8

# Tote 검출 보정은 이 Head 카메라 모드에서 수행했다. AR은 주입받은 카메라의 실제 intrinsic을 사용한다.
HEAD_CAM_WIDTH = 640
HEAD_CAM_HEIGHT = 480
HEAD_CAM_FPS = 30

DEFAULT_GRIPPER_TARGET = 0.80
DEFAULT_GRIPPER_TORQUE = 0.20

# Tote vision과 grasp pose는 이 torso 기준으로 맞춰져 있으므로 프로그램 시작 시 한 번 정확히 고정한다.
INITIAL_TORSO = np.deg2rad([0.0, 30.0, -50.0, 30.0, 0.0, 0.0]).tolist()


# -----------------------------------------------------------------------------
# Tote 파지 자세
# torso는 BEFORE / GRASP / UP 동안 동일하게 유지하며, 아래 값 하나만 사용한다.
# -----------------------------------------------------------------------------

BEFORE_RIGHT = np.deg2rad([-38.23, -53.19, -21.31, -48.14, -63.73, 81.18, 2.39]).tolist()
BEFORE_LEFT = np.deg2rad([-38.23, 53.19, 21.31, -48.14, 63.73, 81.18, -2.39]).tolist()

AFTER_RIGHT = np.deg2rad([-51.651, -35.387, -16.519, -42.941, -31.167, 73.404, 0.001]).tolist()
AFTER_LEFT = np.deg2rad([-51.625, 37.742, 19.947, -44.127, 35.084, 75.497, -0.033]).tolist()

DOWN_RIGHT = np.deg2rad([-53.243, -27.593, -16.509, -45.481, -31.781, 73.370, 0.012]).tolist()
DOWN_LEFT = np.deg2rad([-51.643, 29.044, 19.947, -45.832, 35.513, 75.498, -0.036]).tolist()

STRETCH_RIGHT = np.deg2rad([-34.28, -35.32, -21.87, -68.29, -66.50, 90.79, -12.55]).tolist()
STRETCH_LEFT = np.deg2rad([-34.28, 35.32, 21.87, -68.29, 66.50, 90.79, 12.55]).tolist()

GRASP_RIGHT = np.deg2rad([-37.43, -32.30, -21.34, -49.22, -63.95, 81.79, 2.40]).tolist()
GRASP_LEFT = np.deg2rad([-37.43, 32.30, 21.34, -49.22, 63.95, 81.79, -2.40]).tolist()

UP_RIGHT = np.deg2rad([-4.50, -28.21, -33.62, -106.81, -74.26, 99.28, -19.51]).tolist()
UP_LEFT = np.deg2rad([-4.50, 28.21, 33.62, -106.81, 74.26, 99.28, 19.52]).tolist()

BACK_RIGHT = np.deg2rad([-17.36, -31.32, -35.09, -99.56, -59.69, 98.00, -13.33]).tolist()
BACK_LEFT = np.deg2rad([-17.36, 31.32, 35.09, -99.56, 59.69, 98.00, 13.33]).tolist()

HEAD_DOWN = np.deg2rad([0.0, 43.0]).tolist()    # Tote 인식 / 복귀 자세
HEAD_FORWARD = np.deg2rad([0.0, 0.0]).tolist()  # 정면 AR 마커 인식 자세
HEAD_MOVE_TIME = 2.0
ARM_UP_MOVE_TIME = 2.0



# -----------------------------------------------------------------------------
# 모바일 경로
# target = (x [m], y [m], yaw [rad])
# 아래 target 값만 수정하면 실제 실행 로그도 현재 값에 맞춰 자동으로 바뀐다.
# -----------------------------------------------------------------------------


def compose_relative_targets(*targets):
    """여러 body-frame 상대 target을 하나의 최종 상대 target으로 합성한다."""
    x_m = 0.0
    y_m = 0.0
    yaw_rad = 0.0

    for dx_m, dy_m, dyaw_rad in targets:
        cosine = math.cos(yaw_rad)
        sine = math.sin(yaw_rad)
        x_m += cosine * dx_m - sine * dy_m
        y_m += sine * dx_m + cosine * dy_m
        yaw_rad += dyaw_rad

    return x_m, y_m, yaw_rad

BACK_TARGET = (-0.10, 0.0, 0.0)
TURN_TARGET = (-0.05, -0.05, math.radians(-180.43))
STRAIGHT_TARGET = (0.65, 0.0, 0.0)

OUTBOUND_DIRECT_TARGET = compose_relative_targets(BACK_TARGET, TURN_TARGET, STRAIGHT_TARGET)
OUTBOUND_DIRECT_DURATION = 8.0
OUTBOUND_HEAD_DELAY = 3.0

RETURN_BACK_TARGET = (-0.35, 0.0, 0.0)
RETURN_TURN_TARGET = (0.0, 0.0, math.radians(183.43))
RETURN_STRAIGHT_TARGET = (1.00, 0.0, 0.0)

RETURN_TURN_AND_STRAIGHT_TARGET = compose_relative_targets(RETURN_TURN_TARGET, RETURN_STRAIGHT_TARGET)
RETURN_TURN_AND_STRAIGHT_DURATION = 9.0

# ------------------------------------------------------------------
###############           각 액션 정의             #################
# ------------------------------------------------------------------


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


def move_head_async(robot, head_pose, description: str, delay: float = 0.0) -> Future:
    """Torso 기준을 유지하며 Head 명령을 선택적으로 지연해 주행과 겹친다."""
    future = Future()

    def worker():
        try:
            if delay > 0.0:
                time.sleep(delay)

            if not future.set_running_or_notify_cancel():
                return

            success = move_torso_and_head(robot, INITIAL_TORSO, head_pose, minimum_time=HEAD_MOVE_TIME)
            future.set_result(success)
        except Exception as error:  # SDK 명령 예외는 메인 시퀀스에서 처리할 수 있도록 Future로 전달한다.
            future.set_exception(error)

    print(description)
    threading.Thread(target=worker, name="head-pose", daemon=True).start()
    return future


def wait_for_head_move(future: Future, description: str) -> bool:
    """동시에 실행한 Head 이동 결과를 다음 카메라 인식 또는 사이클 시작 전에 확인한다."""
    try:
        if future.result():
            return True
    except Exception as error:
        print(f"{description} 예외: {error}")
        return False

    print(f"{description} 실패")
    return False


def finish_pending_head_move(future: Future | None, description: str) -> None:
    """모바일 이동 중 예외가 나더라도 이미 시작한 Head 명령이 끝날 때까지 정리한다."""
    if future is not None:
        if future.cancel():
            print(f"{description} 예약 취소")
        else:
            wait_for_head_move(future, description)


def move_arms_async(robot, right_arm, left_arm, description: str) -> Future:
    """양팔 명령을 별도 thread에서 실행해 모바일 회전과 겹친다."""
    future = Future()

    def worker():
        try:
            success = move_both_arms(robot, right_arm, left_arm, minimum_time=ARM_UP_MOVE_TIME)
            future.set_result(success)
        except Exception as error:  # SDK 명령 예외는 메인 시퀀스에서 처리할 수 있도록 Future로 전달한다.
            future.set_exception(error)

    print(description)
    threading.Thread(target=worker, name="arm-pose", daemon=True).start()
    return future


def wait_for_arm_move(future: Future, description: str) -> bool:
    """동시에 실행한 양팔 이동이 끝났는지 다음 주행 구간 전에 확인한다."""
    try:
        if future.result():
            return True
    except Exception as error:
        print(f"{description} 예외: {error}")
        return False

    print(f"{description} 실패")
    return False


def finish_pending_arm_move(future: Future | None, description: str) -> None:
    """복귀 주행 중 예외가 나더라도 이미 시작한 양팔 명령이 끝날 때까지 정리한다."""
    if future is not None:
        wait_for_arm_move(future, description)


def run_turn_and_go(robot, monitor) -> bool:
    """BACK + TURN + STRAIGHT direct target으로 이송하고 기존 직진 시점에 Head를 든다."""
    stream = robot.create_command_stream(priority=10)
    head_move = None

    try:
        head_move = move_head_async(
            robot,
            HEAD_FORWARD,
            f"이송 시작 {OUTBOUND_HEAD_DELAY:.1f}초 후 정면 AR 인식을 위해 Head 들기",
            delay=OUTBOUND_HEAD_DELAY,
        )
        route_ok = run_mobile_leg(
            robot,
            monitor,
            stream,
            "이송 direct target",
            OUTBOUND_DIRECT_TARGET,
            OUTBOUND_DIRECT_DURATION,
            True,
            0.2,
        )
        head_ok = wait_for_head_move(head_move, "Head 정면 자세 이동")
        head_move = None

        return route_ok and head_ok

    finally:
        finish_pending_head_move(head_move, "Head 정면 자세 이동")
        stream.cancel()
        stream.wait_for(500)


def run_return_route(robot, monitor) -> bool:
    """BACK을 단독 수행한 뒤 TURN + STRAIGHT direct target으로 복귀한다."""
    stream = robot.create_command_stream(priority=10)
    head_move = None
    arm_up_move = None

    try:
        if not run_mobile_leg(robot, monitor, stream, "복귀 1/2: BACK", RETURN_BACK_TARGET, 3.0, False, 0.0):
            return False

        arm_up_move = move_arms_async(
            robot,
            BACK_RIGHT,
            BACK_LEFT,
            "복귀 2/2과 동시에 양팔을 UP 자세로 이동",
        )
        route_ok = run_mobile_leg(
            robot,
            monitor,
            stream,
            "복귀 2/2: TURN + STRAIGHT direct target",
            RETURN_TURN_AND_STRAIGHT_TARGET,
            RETURN_TURN_AND_STRAIGHT_DURATION,
            True,
            0.2,
        )

        head_move = move_head_async(robot, HEAD_DOWN, "복귀 1/3과 동시에 다음 Tote 인식을 위해 Head 숙이기")

        head_ok = wait_for_head_move(head_move, "Head Tote 인식 자세 이동")
        head_move = None
        arm_up_ok = True if arm_up_move is None else wait_for_arm_move(arm_up_move, "양팔 UP 자세 이동")
        arm_up_move = None
        return route_ok and head_ok and arm_up_ok

    finally:
        finish_pending_head_move(head_move, "Head Tote 인식 자세 이동")
        finish_pending_arm_move(arm_up_move, "양팔 UP 자세 이동")
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
    """Tote를 정렬한 뒤 양팔을 BEFORE → GRASP → UP 순서로 이동해 파지한다."""


    print("[2/4] 현재 자세에서 Tote 영상 인식 + one-shot 정렬")
    if not tote_aligner.align(robot, monitor, verify=True):
        print("Tote one-shot 정렬 실패")
        return False

    print("[1/4] 현재 자세 → BEFORE")
    if not move_both_arms(robot, BEFORE_RIGHT, BEFORE_LEFT, minimum_time=1.0):
        print("BEFORE 자세 이동 실패")
        return False

    print("[3/4] BEFORE → GRASP")
    if not move_both_arms(robot, GRASP_RIGHT, GRASP_LEFT, minimum_time=1.5):
        print("GRASP 자세 이동 실패")
        return False

    print(f"그리퍼 닫기: target={gripper_target:.2f}, torque={gripper_torque:.2f} Nm")
    gripper.close(target=gripper_target, torque=gripper_torque, duration=0.8)
    print(f"그리퍼 현재 위치: {gripper.get_positions().round(3)}")

    print("[4/4] GRASP → UP")
    if not move_both_arms(robot, UP_RIGHT, UP_LEFT, minimum_time=1.2):
        print("UP 자세 이동 실패")
        return False

    return True


def lower_release_and_retract(robot, gripper) -> bool:
    """AR 정렬 후 UP에서 DOWN로 내려놓고 그리퍼를 연 뒤, 손잡이에서 빠져나오도록 AFTER 자세로 양팔을 후퇴한다."""
    print("[1/6] UP → stretch")
    if not move_both_arms(robot, STRETCH_RIGHT, STRETCH_LEFT, minimum_time=1.0):
        print("STRETCH 자세 이동 실패")
        return False

    print("[1/6] stretch -> Down ")
    if not move_both_arms(robot, DOWN_RIGHT, DOWN_LEFT, minimum_time=1.0):
        print("DOWN 자세 이동 실패")
        return False

    print("그리퍼 열기")
    gripper.open(duration=1.0)

    print("DOWN → AFTER")
    if not move_both_arms(robot, AFTER_RIGHT, AFTER_LEFT, minimum_time=1.0):
        print("AFTER 자세 이동 실패")
        return False

    return True



# -------------------------------------------------------------------
###############           반복 시퀀스 정의            #################
# -------------------------------------------------------------------



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

    print(f"이송 시작: BACK + TURN + STRAIGHT direct {describe_target(OUTBOUND_DIRECT_TARGET)}")
    if not run_turn_and_go(robot, monitor):
        raise RuntimeError("이송 direct target 주행 실패")

    print("AR 마커 one-shot 정렬")
    if not ar_aligner.align(robot, monitor):
        raise RuntimeError("AR 마커 정렬 실패")

    if not lower_release_and_retract(robot, gripper):
        raise RuntimeError("Tote 배치 실패")

    print(
        f"복귀 시작: BACK {describe_target(RETURN_BACK_TARGET)} → "
        f"TURN + STRAIGHT direct {describe_target(RETURN_TURN_AND_STRAIGHT_TARGET)}"
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
    parser.add_argument(
        "--camera-serial",
        "--tote-camera-serial",
        dest="camera_serial",
        default=HEAD_CAMERA_SERIAL,
        help="Tote/AR 인식에 공용으로 사용할 Head RealSense serial",
    )
    parser.add_argument("--show-tote", action="store_true", help="Tote 검출 OpenCV 화면 표시")
    parser.add_argument("--gripper-target", type=float, default=DEFAULT_GRIPPER_TARGET, help="그리퍼 닫힘 위치")
    parser.add_argument("--gripper-torque", type=float, default=DEFAULT_GRIPPER_TORQUE, help="그리퍼 파지 토크 [Nm]")
    parser.add_argument("--cycles", type=int, default=0, help="반복 횟수, 0이면 Ctrl+C 전까지 무한 반복")
    args = parser.parse_args()

    robot = initialize_mobile(args.address, args.model, power=".*", servo=".*", unlimited=False)

    gripper = None
    head_camera = RealSenseCamera(HEAD_CAM_WIDTH, HEAD_CAM_HEIGHT, HEAD_CAM_FPS, serial=args.camera_serial)
    tote_aligner = ToteAligner(camera=head_camera, show=args.show_tote)
    ar_aligner = ARAligner(marker_id=args.marker_id, camera=head_camera)
    monitor = OdometryMonitor()
    wcs_publisher = WcsPublisher(robot_model=robot.model())
    state_update_started = False
    wcs_publisher_started = False
    head_camera_started = False
    completed_cycles = 0

    try:
        # 초기 상태를 먼저 설정한다. RobotState가 들어오면 publisher가 최신 상태와 함께 WCS로 전송한다.
        wcs_publisher.set_work_state("IDLE")
        wcs_publisher.start()
        wcs_publisher_started = True

        robot.set_tool_flange_output_voltage("right", 12)
        robot.set_tool_flange_output_voltage("left", 12)
        time.sleep(0.5)

        gripper = GripperController(position_torque=args.gripper_torque)
        gripper.connect()
        gripper.open(duration=2.0)

        # SDK state 구독은 한 번만 시작하고, 같은 state를 오도메트리와 WCS 상태 저장부에 함께 전달한다.
        def on_robot_state(state, *callback_args):
            monitor.on_state(state, *callback_args)
            wcs_publisher.on_state(state, *callback_args)

        robot.start_state_update(on_robot_state, rate=50)
        state_update_started = True

        if not wait_for_odometry(monitor):
            raise RuntimeError("Odometry를 받지 못했습니다.")

        # 초기에는 Tote 시야를 위한 Torso / Head만 기준 자세로 맞춘다. 양팔 BEFORE 이동은 Tote 정렬 후에 수행한다.
        initial_head_move = move_head_async(robot, HEAD_DOWN, "초기 Torso / Head 기준 자세로 이동")
        if not wait_for_head_move(initial_head_move, "초기 Torso / Head 자세 이동"):
            raise RuntimeError("초기 Torso / Head 자세 이동 실패")

        print(
            f"Head 공용 카메라 시작: serial={args.camera_serial}, "
            f"{HEAD_CAM_WIDTH}x{HEAD_CAM_HEIGHT}@{HEAD_CAM_FPS}"
        )
        head_camera.start()
        head_camera_started = True

        cycle_index = 1

        while args.cycles == 0 or cycle_index <= args.cycles:
            wcs_publisher.set_work_state("WORKING")
            run_cycle(robot, monitor, gripper, tote_aligner, ar_aligner, args, cycle_index)
            wcs_publisher.set_work_state("DONE")
            completed_cycles += 1
            cycle_index += 1

        print(f"요청한 {completed_cycles}개 사이클 완료")

    except KeyboardInterrupt:
        wcs_publisher.set_work_state("IDLE")
        print(f"\n사용자가 반복 데모를 중단했습니다. 완료 사이클: {completed_cycles}")

    except Exception as error:
        wcs_publisher.set_work_state("ERROR", str(error))
        print(f"반복 데모 실패: {error} | 완료 사이클: {completed_cycles}")

    finally:
        # DONE / ERROR처럼 짧게 유지될 수 있는 마지막 상태를 즉시 한 번 전송한 뒤 publisher를 종료한다.
        if wcs_publisher_started:
            try:
                wcs_publisher.stop(flush=True)
            except Exception as error:
                print(f"WCS publisher 종료 실패: {error}")

        if head_camera_started:
            try:
                head_camera.stop()
            except Exception as error:
                print(f"Head 공용 카메라 종료 실패: {error}")

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