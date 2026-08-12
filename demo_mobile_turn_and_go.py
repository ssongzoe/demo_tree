#!/usr/bin/env python3
"""RB-Y1 모바일 회전 후 직진 데모.

사용 모듈
- control.mobile_controller
    - 모바일 초기화
    - Odometry
    - SE(2) 주행
"""

import math

from control.mobile_controller import (
    OdometryMonitor, 
    build_leg, 
    initialize_mobile, 
    move_leg, 
    odom_pose, 
    wait_for_odometry,
)


ADDRESS = "192.168.30.1:50051"


# LEG 1: 현재 자세 기준으로 회전하면서 이동
TURN_TARGET = (
    -0.50,                  # x [m]
    -0.05,                  # y [m]
    math.radians(-179.43),  # theta [rad]
)

# LEG 2: LEG 1 종료 자세 기준으로 앞으로 직진
STRAIGHT_TARGET = (
    0.85,  # forward [m]
    0.0,   # lateral [m]
    0.0,   # rotation [rad]
)


def main():
    # 모바일 주행에 필요한 wheel servo만 활성화한다.
    robot = initialize_mobile(address=ADDRESS, model="m")
    monitor = OdometryMonitor()

    try:
        # 모바일 위치 / 자세 확인을 위해 odometry 수신 시작
        robot.start_state_update(monitor.on_state, rate=50)

        if not wait_for_odometry(monitor):
            print("Odometry를 받지 못했습니다.")
            return

        # 두 LEG를 끊김 없이 연결하기 위해 같은 stream 사용
        stream = robot.create_command_stream(priority=10)

        try:
            # LEG 1: 회전 + 이동
            leg1 = build_leg(
                start=odom_pose(monitor.odom),
                target=TURN_TARGET,
                absolute=False,
                duration=10.0,
                turn_direction="shortest",
            )

            print("LEG 1: 회전 + 이동")

            if not move_leg(robot, monitor, leg1, settle=0.0, stream=stream, stop_at_end=False):
                print("LEG 1 실패")
                return

            # LEG 2: 직진
            leg2 = build_leg(
                start=odom_pose(monitor.odom),
                target=STRAIGHT_TARGET,
                absolute=False,
                duration=5.0,
                turn_direction="shortest",
            )

            print("LEG 2: 직진")

            if not move_leg(robot, monitor, leg2, settle=1.5, stream=stream, stop_at_end=True):
                print("LEG 2 실패")
                return

            print("데모 완료")

        finally:
            stream.cancel()
            stream.wait_for(500)

    finally:
        try:
            robot.stop_state_update()
        except Exception:
            pass

        robot.disconnect()


if __name__ == "__main__":
    main()
