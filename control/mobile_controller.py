#!/usr/bin/env python3
"""RB-Y1 모바일 베이스 제어 모듈.

rby1_sdk에서 사용하는 주요 제어
- rby.create_robot(): 로봇 객체 생성
- power_on(), servo_on(): 모바일 주행에 필요한 전원 / wheel servo 활성화
- enable_control_manager(): 제어 매니저 활성화
- robot.start_state_update(): state.odometry 수신
- SE2VelocityCommandBuilder: vx, vy, yaw 속도 명령 생성
- robot.create_command_stream(): 모바일 속도 명령 연속 전송
"""

from dataclasses import dataclass
import math
import time

import rby1_sdk as rby


# ============================================================
# 모바일 제어 설정
# ============================================================

CTRL_HZ = 30
DT = 1.0 / CTRL_HZ

LINEAR_ACC_LIMIT = 2.0
ANGULAR_ACC_LIMIT = 2.5

CONTROL_HOLD_TIME = 1.0
SEND_TIMEOUT_MS = 2000

KP_LIN = 0.8
KP_ANG = 1.2

MAX_LIN_VEL = 0.8
MAX_ANG_VEL = 1.5

POS_TOL = 0.02
ANG_TOL = math.radians(1.5)


# ============================================================
# 모바일 초기화
# ============================================================

def initialize_mobile(
    address,
    model="m",
    power="48v",
    servo="^(wheel_.*|right_wheel|left_wheel)$",
    unlimited=True,
):
    """모바일 주행에 필요한 로봇 연결 및 wheel servo를 준비한다."""
    robot = rby.create_robot(address, model)

    if not robot.connect():
        raise ConnectionError(f"로봇 연결 실패: {address}")

    if not robot.is_power_on(power) and not robot.power_on(power):
        raise RuntimeError("로봇 전원 활성화 실패")

    if not robot.is_servo_on(servo) and not robot.servo_on(servo):
        raise RuntimeError("모바일 wheel servo 활성화 실패")

    state = robot.get_control_manager_state().state

    if state in (rby.ControlManagerState.State.MajorFault, rby.ControlManagerState.State.MinorFault):
        if not robot.reset_fault_control_manager():
            raise RuntimeError("제어 매니저 fault 초기화 실패")

    state = robot.get_control_manager_state().state

    if state != rby.ControlManagerState.State.Enabled:
        if not robot.enable_control_manager(unlimited_mode_enabled=unlimited):
            raise RuntimeError("제어 매니저 활성화 실패")

    return robot


# ============================================================
# Odometry
# ============================================================

@dataclass
class Leg:
    """하나의 SE(2) 주행 구간."""

    x0: float
    y0: float
    th0: float
    x1: float
    y1: float
    dth: float
    duration: float


class OdometryMonitor:
    """RB-Y1 state update에서 odometry를 보관한다."""

    def __init__(self):
        self.odom = None

    def on_state(self, state, _control_manager):
        self.odom = state.odometry


def wait_for_odometry(monitor, timeout_s=3.0) -> bool:
    """첫 odometry가 수신될 때까지 기다린다."""
    deadline = time.monotonic() + timeout_s

    while monitor.odom is None and time.monotonic() < deadline:
        time.sleep(0.05)

    return monitor.odom is not None


def wrap_angle(angle: float) -> float:
    """각도를 -pi ~ pi 범위로 정규화한다."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def odom_pose(odom):
    """4x4 odometry 행렬에서 x, y, yaw를 추출한다."""
    return (
        float(odom[0, 2]),
        float(odom[1, 2]),
        float(math.atan2(odom[1, 0], odom[0, 0])),
    )


def compose_pose(base, relative):
    """현재 자세 기준 상대 이동량을 odom 좌표계 목표점으로 변환한다."""
    x0, y0, th0 = base
    dx, dy, dth = relative

    c = math.cos(th0)
    s = math.sin(th0)

    return (
        x0 + c * dx - s * dy,
        y0 + s * dx + c * dy,
        th0 + dth,
    )


# ============================================================
# Trajectory
# ============================================================

def build_leg(start, target, absolute, duration, turn_direction):
    """상대 좌표 또는 odom 절대 좌표 기준으로 하나의 주행 구간을 만든다."""
    x0, y0, th0 = start

    if absolute:
        x1, y1, target_heading = target
        dth = wrap_angle(target_heading - th0)

        if turn_direction == "cw" and dth > 0.0:
            dth -= 2.0 * math.pi
        elif turn_direction == "ccw" and dth < 0.0:
            dth += 2.0 * math.pi
    else:
        x1, y1, _ = compose_pose(start, target)
        dth = target[2]

    return Leg(
        x0=x0,
        y0=y0,
        th0=th0,
        x1=x1,
        y1=y1,
        dth=dth,
        duration=duration,
    )


def quintic(tau):
    """시작과 끝이 부드러운 5차 trajectory를 만든다."""
    tau = min(1.0, max(0.0, tau))

    position = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    velocity = 30.0 * tau**2 * (1.0 - tau) ** 2

    return position, velocity


# ============================================================
# rby1_sdk SE(2) 명령
# ============================================================

def build_se2_command(vx, vy, w):
    """모바일 베이스 SE(2) 속도 명령을 생성한다."""
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_mobility_command(
            rby.SE2VelocityCommandBuilder()
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(CONTROL_HOLD_TIME))
            .set_minimum_time(DT)
            .set_velocity([float(vx), float(vy)], float(w))
            .set_acceleration_limit([LINEAR_ACC_LIMIT, LINEAR_ACC_LIMIT], ANGULAR_ACC_LIMIT)
        )
    )


# ============================================================
# 주행
# ============================================================

def move_leg(robot, monitor, leg, settle, *, stream=None, stop_at_end=True) -> bool:
    """Odometry feedback으로 하나의 SE(2) trajectory를 추종한다."""
    dx = leg.x1 - leg.x0
    dy = leg.y1 - leg.y0
    th_end = leg.th0 + leg.dth

    # 외부 stream을 받으면 다음 leg에서도 이어서 사용한다.
    owns_stream = stream is None

    if owns_stream:
        stream = robot.create_command_stream(priority=10)

    t0 = time.monotonic()
    tick = 0
    stream_failed = False

    try:
        while True:
            t = time.monotonic() - t0
            x, y, th = odom_pose(monitor.odom)

            # 현재 trajectory 진행률
            s, sp = quintic(t / leg.duration)
            dsdt = sp / leg.duration

            # 목표 자세
            xd = leg.x0 + dx * s
            yd = leg.y0 + dy * s
            thd = leg.th0 + leg.dth * s

            # Feed-forward 속도
            vx_world = dx * dsdt
            vy_world = dy * dsdt
            w_ff = leg.dth * dsdt

            # 현재 위치 오차
            ex_world = xd - x
            ey_world = yd - y

            c = math.cos(th)
            sn = math.sin(th)

            # odom 좌표계 오차를 robot body 좌표계로 변환
            ex_body = c * ex_world + sn * ey_world
            ey_body = -sn * ex_world + c * ey_world
            heading_error = wrap_angle(thd - th)

            # 최종 목표점 기준 오차
            pos_error = math.hypot(leg.x1 - x, leg.y1 - y)
            final_heading_error = wrap_angle(th_end - th)

            # 설정된 주행 시간이 지난 뒤 목표 오차를 확인한다.
            if t >= leg.duration:
                arrived = pos_error <= POS_TOL and abs(final_heading_error) <= ANG_TOL

                if arrived or t >= leg.duration + settle:
                    break

            # Feed-forward + feedback
            vx = c * vx_world + sn * vy_world + KP_LIN * ex_body
            vy = -sn * vx_world + c * vy_world + KP_LIN * ey_body
            w = w_ff + KP_ANG * heading_error

            # 속도 제한
            vx = max(-MAX_LIN_VEL, min(MAX_LIN_VEL, vx))
            vy = max(-MAX_LIN_VEL, min(MAX_LIN_VEL, vy))
            w = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, w))

            feedback = stream.send_command(build_se2_command(vx, vy, w), SEND_TIMEOUT_MS)

            if feedback.status == rby.RobotCommandFeedback.Status.Finished:
                stream_failed = True
                break

            # 30 Hz 제어 주기 유지
            tick += 1
            next_tick = t0 + tick * DT
            sleep_time = next_tick - time.monotonic()

            if sleep_time > 0:
                time.sleep(sleep_time)

        # 마지막 leg에서만 0 속도를 보내 정지한다.
        if not stream_failed and stop_at_end:
            for _ in range(3):
                stream.send_command(build_se2_command(0.0, 0.0, 0.0), SEND_TIMEOUT_MS)
                time.sleep(DT)

            time.sleep(0.3)

    finally:
        # 함수 내부에서 만든 stream만 직접 종료한다.
        if owns_stream:
            stream.cancel()
            stream.wait_for(500)

    x, y, th = odom_pose(monitor.odom)

    print(f"주행 종료: x={x:+.3f}, y={y:+.3f}, heading={math.degrees(th):+.1f} deg")

    return not stream_failed
