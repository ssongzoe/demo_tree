#!/usr/bin/env python3
"""RB-Y1 모바일 베이스의 direct / waypoints 주행 경로 비교 테스트.

현재 odometry 자세를 시작점으로 아래 세 상대 target을 실행한다.
- BACK_TARGET
- TURN_TARGET
- STRAIGHT_TARGET

실행 방식
- direct: 세 target을 합성한 최종 pose 하나를 5차 trajectory로 추종
- waypoints: 세 target의 5차 profile을 겹쳐 중간 정지 없이 연속 추종


python demos/demo_mobile_route_test.py --route-mode direct
python demos/demo_mobile_route_test.py --route-mode waypoints


"""

import argparse
import math

from control.mobile_controller import (
    OdometryMonitor,
    build_leg,
    build_route,
    compose_relative_targets,
    initialize_mobile,
    move_leg,
    move_route,
    odom_pose,
    wait_for_odometry,
)


ADDRESS = "192.168.30.1:50051"

BACK_TARGET = (-0.10, 0.0, 0.0)
TURN_TARGET = (-0.05, -0.05, math.radians(-180.43))
STRAIGHT_TARGET = (0.65, 0.0, 0.0)

TARGETS = (BACK_TARGET, TURN_TARGET, STRAIGHT_TARGET)
DURATIONS = (2.0, 7.0, 5.0)
DIRECT_TARGET = compose_relative_targets(*TARGETS)

DEFAULT_ROUTE_BLEND = 0.75
SETTLE_TIME = 0.2


def describe_target(target) -> str:
    """target의 x/y/yaw를 로그 문자열로 변환한다."""
    x_m, y_m, yaw_rad = target
    return f"x={x_m:+.3f} m, y={y_m:+.3f} m, yaw={math.degrees(yaw_rad):+.2f} deg"


def run_direct(robot, monitor, duration: float) -> bool:
    """세 상대 target을 합성한 최종 pose 하나로 이동한다."""
    print(f"[direct] 최종 target: {describe_target(DIRECT_TARGET)}")
    print(f"[direct] duration: {duration:.2f} s")

    leg = build_leg(
        start=odom_pose(monitor.odom),
        target=DIRECT_TARGET,
        absolute=False,
        duration=duration,
        turn_direction="shortest",
    )

    return move_leg(robot, monitor, leg, settle=SETTLE_TIME, stop_at_end=True)


def run_waypoints(robot, monitor, blend_time: float) -> bool:
    """세 target의 profile을 겹쳐 BACK → TURN → STRAIGHT 경로를 연속 추종한다."""
    print(f"[waypoints] BACK:     {describe_target(BACK_TARGET)}")
    print(f"[waypoints] TURN:     {describe_target(TURN_TARGET)}")
    print(f"[waypoints] STRAIGHT: {describe_target(STRAIGHT_TARGET)}")
    print(f"[waypoints] blend: {blend_time:.2f} s")

    route = build_route(
        start=odom_pose(monitor.odom),
        targets=TARGETS,
        durations=DURATIONS,
        absolute=False,
    )

    return move_route(
        robot,
        monitor,
        route,
        settle=SETTLE_TIME,
        blend_time=blend_time,
        stop_at_end=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RB-Y1 모바일 direct / waypoints 경로 비교")
    parser.add_argument("--address", default=ADDRESS, help="로봇 주소")
    parser.add_argument("--model", choices=("a", "m"), default="m", help="RB-Y1 모델")
    parser.add_argument(
        "--route-mode",
        choices=("direct", "waypoints"),
        required=True,
        help="direct=최종 pose 한 번, waypoints=BACK/TURN/STRAIGHT 연속 trajectory",
    )
    parser.add_argument(
        "--route-blend",
        type=float,
        default=DEFAULT_ROUTE_BLEND,
        help="waypoints에서 인접 구간을 겹치는 시간 [s]",
    )
    args = parser.parse_args()

    if not 0.0 <= args.route_blend < min(DURATIONS):
        parser.error(f"--route-blend는 0.0 이상 {min(DURATIONS):.1f} 미만이어야 합니다.")

    # direct와 waypoints의 전체 실행 시간을 같게 맞춰 경로 형태만 비교하기 쉽게 한다.
    comparison_duration = sum(DURATIONS) - args.route_blend * (len(DURATIONS) - 1)

    robot = None
    state_update_started = False

    try:
        robot = initialize_mobile(args.address, args.model, unlimited=False)
        monitor = OdometryMonitor()

        robot.start_state_update(monitor.on_state, rate=50)
        state_update_started = True

        if not wait_for_odometry(monitor):
            raise RuntimeError("Odometry를 받지 못했습니다.")

        print(f"현재 자세: {describe_target(odom_pose(monitor.odom))}")
        print(f"테스트 모드: {args.route_mode}")

        if args.route_mode == "direct":
            success = run_direct(robot, monitor, comparison_duration)
        else:
            success = run_waypoints(robot, monitor, args.route_blend)

        if not success:
            raise RuntimeError(f"모바일 경로 테스트 실패: route_mode={args.route_mode}")

        print(f"테스트 완료: route_mode={args.route_mode}")

    finally:
        if robot is not None:
            if state_update_started:
                try:
                    robot.stop_state_update()
                except Exception:
                    pass

            try:
                robot.disable_control_manager()
            except Exception:
                pass

            robot.disconnect()
            print("모바일 제어와 연결을 정리했습니다.")


if __name__ == "__main__":
    main()
