#!/usr/bin/env python3
"""RB-Y1 0.80 m 직진 후 AR 마커 정렬 데모.

동작
1. x 방향 0.80 m 직진
2. AR 마커 yaw 정렬
3. AR 마커 전후 / 좌우 위치 정렬
"""

from control.mobile_controller import OdometryMonitor, build_leg, initialize_mobile, move_leg, odom_pose, wait_for_odometry
from skills.ar_align import align_to_marker


ADDRESS = "192.168.30.1:50051"

DRIVE_TARGET = (
    0.80,  # x [m]
    0.0,   # y [m]
    0.0,   # theta [rad]
)

DRIVE_DURATION = 5.0

MARKER_ID = 7

# 여러 RealSense가 연결되어 있다면 D405 시리얼을 문자열로 지정한다.
CAMERA_SERIAL = None


def main():
    robot = initialize_mobile(address=ADDRESS, model="m")
    monitor = OdometryMonitor()

    try:
        robot.start_state_update(monitor.on_state, rate=50)

        if not wait_for_odometry(monitor):
            print("Odometry를 받지 못했습니다.")
            return

        # 1단계: 현재 자세 기준 x 방향 0.80 m 직진
        drive_leg = build_leg(
            start=odom_pose(monitor.odom),
            target=DRIVE_TARGET,
            absolute=False,
            duration=DRIVE_DURATION,
            turn_direction="shortest",
        )

        print("STEP 1: 0.80 m 직진")

        if not move_leg(robot, monitor, drive_leg, settle=1.0):
            print("직진 실패")
            return

        # 2단계: AR 마커를 기준으로 yaw와 위치를 정렬
        print("STEP 2: AR 마커 정렬")

        if not align_to_marker(
            robot,
            monitor,
            marker_id=MARKER_ID,
            camera_serial=CAMERA_SERIAL,
        ):
            print("AR 정렬 실패")
            return

        print("데모 완료")

    finally:
        try:
            robot.stop_state_update()
        except Exception:
            pass

        robot.disconnect()


if __name__ == "__main__":
    main()
