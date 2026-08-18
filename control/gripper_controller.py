#!/usr/bin/env python3
"""RB-Y1 양손 Dynamixel 그리퍼 제어기.

정규화 위치는 양쪽 모두 0.0=완전 열림, 1.0=완전 닫힘이다.
연결 시 기구 한계를 찾기 위해 양방향으로 약 3초씩 홈 동작을 수행한다.
홈 동작 전에는 그리퍼 사이를 반드시 비워야 한다.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import rby1_sdk as rby
import rby1_sdk.upc as upc


logger = logging.getLogger(__name__)

GRIPPER_BAUD_RATE = 2_000_000
GRIPPER_IDS = [0, 1]
GRIPPER_HOMING_TORQUE = 0.46
GRIPPER_HOMING_STEPS = 30
GRIPPER_MAX_POSITION_TORQUE = 0.46


class GripperController:
    """`/dev/rby1_gripper`에 연결된 양손 그리퍼를 제어한다."""

    def __init__(self, position_torque: float = 0.20) -> None:
        self._bus = None
        self._open_rad = np.zeros(2, dtype=np.float64)
        self._close_rad = np.zeros(2, dtype=np.float64)
        self._homed = False
        self._position_torque = self._validate_torque(position_torque)

    @staticmethod
    def _validate_torque(torque: float) -> float:
        torque = float(torque)
        if not 0.0 < torque <= GRIPPER_MAX_POSITION_TORQUE:
            raise ValueError(
                f"position_torque는 0보다 크고 {GRIPPER_MAX_POSITION_TORQUE:.2f} Nm 이하여야 합니다."
            )
        return torque

    def connect(self) -> None:
        """통신을 연결하고 양방향 홈 동작으로 열림/닫힘 범위를 찾는다."""
        self._bus = rby.DynamixelBus(upc.GripperDeviceName)

        if not self._bus.open_port():
            raise RuntimeError(f"그리퍼 포트를 열지 못했습니다: {upc.GripperDeviceName}")

        upc.initialize_device(upc.GripperDeviceName)

        if not self._bus.set_baud_rate(GRIPPER_BAUD_RATE):
            raise RuntimeError(f"그리퍼 baud rate 설정 실패: {GRIPPER_BAUD_RATE}")

        self._bus.set_torque_constant([1.0, 1.0])

        for dev_id in GRIPPER_IDS:
            if not self._bus.ping(dev_id):
                raise RuntimeError(f"그리퍼 모터가 응답하지 않습니다: ID={dev_id}")

        self._home()
        logger.info("그리퍼 연결 및 홈 동작 완료")

    def disconnect(self) -> None:
        """그리퍼 토크를 끄고 연결 상태를 초기화한다."""
        if self._bus is not None:
            try:
                self._bus.group_sync_write_torque_enable(
                    [(dev_id, rby.DynamixelBus.TorqueDisable) for dev_id in GRIPPER_IDS]
                )
            except Exception:
                pass

        self._bus = None
        self._homed = False

    def _set_operating_mode(self, mode: int) -> None:
        """양쪽 모터의 Dynamixel 동작 모드를 변경한다."""
        self._bus.group_sync_write_torque_enable(
            [(dev_id, rby.DynamixelBus.TorqueDisable) for dev_id in GRIPPER_IDS]
        )
        self._bus.group_sync_write_operating_mode([(dev_id, mode) for dev_id in GRIPPER_IDS])
        self._bus.group_sync_write_torque_enable(
            [(dev_id, rby.DynamixelBus.TorqueEnable) for dev_id in GRIPPER_IDS]
        )

    def _home(self) -> None:
        """양방향 기구 한계를 측정하여 양쪽 그리퍼의 정규화 범위를 생성한다."""
        print("그리퍼 홈 동작 시작: 양방향으로 약 3초씩 움직입니다.")
        self._set_operating_mode(rby.DynamixelBus.CurrentControlMode)

        q = np.zeros(2, dtype=np.float64)
        min_q = np.full(2, np.inf)
        max_q = np.full(2, -np.inf)

        for direction in range(2):
            torque = GRIPPER_HOMING_TORQUE if direction == 0 else -GRIPPER_HOMING_TORQUE

            for _ in range(GRIPPER_HOMING_STEPS):
                self._bus.group_sync_write_send_torque([(dev_id, torque) for dev_id in GRIPPER_IDS])
                result = self._bus.group_fast_sync_read_encoder(GRIPPER_IDS)

                if result is not None:
                    for dev_id, encoder in result:
                        q[dev_id] = encoder

                min_q = np.minimum(min_q, q)
                max_q = np.maximum(max_q, q)
                time.sleep(0.1)

        self._bus.group_sync_write_send_torque([(dev_id, 0.0) for dev_id in GRIPPER_IDS])
        self._set_operating_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)

        # 실제 그리퍼 장착 방향
        # Right: max encoder = OPEN, min encoder = CLOSED
        # Left:  min encoder = OPEN, max encoder = CLOSED
        self._open_rad = np.array([max_q[0], min_q[1]],dtype=np.float64)
        self._close_rad = np.array([min_q[0], max_q[1]],dtype=np.float64)


        spans = np.abs(self._close_rad - self._open_rad)
        if np.any(spans < 0.01):
            self.disconnect()
            raise RuntimeError(f"그리퍼 홈 범위가 너무 작습니다: spans={spans}")

        self._homed = True
        self.set_position_torque(self._position_torque)

        print(
            f"그리퍼 홈 동작 완료: open={self._open_rad.round(4)}, "
            f"close={self._close_rad.round(4)}"
        )

    def set_position_torque(self, torque: float) -> None:
        """Current-based Position 모드에서 사용할 최대 파지 토크를 설정한다."""
        self._position_torque = self._validate_torque(torque)

        if self._homed:
            self._bus.group_sync_write_send_torque(
                [(dev_id, self._position_torque) for dev_id in GRIPPER_IDS]
            )

        print(f"그리퍼 위치 제어 토크: {self._position_torque:.2f} Nm")

    def set_positions(self, normalized) -> None:
        """양쪽 목표 위치 `[right, left]`를 0.0~1.0 범위로 명령한다."""
        if not self._homed:
            raise RuntimeError("그리퍼가 연결 및 홈 완료 상태가 아닙니다.")

        normalized = np.clip(np.asarray(normalized, dtype=np.float64), 0.0, 1.0)
        if normalized.shape != (2,):
            raise ValueError("그리퍼 목표 위치는 [right, left] 형태여야 합니다.")

        target_rad = self._open_rad + normalized * (self._close_rad - self._open_rad)
        self._bus.group_sync_write_send_position(
            [(dev_id, float(position)) for dev_id, position in zip(GRIPPER_IDS, target_rad)]
        )

    def get_positions(self) -> np.ndarray:
        """현재 양쪽 그리퍼 위치를 `[right, left]` 정규화 값으로 반환한다."""
        if not self._homed:
            return np.zeros(2, dtype=np.float64)

        result = self._bus.group_fast_sync_read_encoder(GRIPPER_IDS)
        if result is None:
            return np.zeros(2, dtype=np.float64)

        encoder_map = {dev_id: encoder for dev_id, encoder in result}
        current_rad = np.array([encoder_map[dev_id] for dev_id in GRIPPER_IDS])
        span = self._close_rad - self._open_rad
        safe_span = np.where(np.abs(span) > 1e-6, span, 1.0)

        return np.clip((current_rad - self._open_rad) / safe_span, 0.0, 1.0)

    def move_positions(self, target, duration: float = 2.0, steps: int = 30) -> None:
        """현재 위치에서 목표 위치까지 smoothstep 보간으로 천천히 이동한다."""
        if duration <= 0.0 or steps < 2:
            raise ValueError("duration은 0보다 커야 하고 steps는 2 이상이어야 합니다.")

        start = self.get_positions()
        target = np.clip(np.asarray(target, dtype=np.float64), 0.0, 1.0)
        if target.shape != (2,):
            raise ValueError("그리퍼 목표 위치는 [right, left] 형태여야 합니다.")

        for ratio in np.linspace(0.0, 1.0, steps):
            smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            self.set_positions(start + smooth_ratio * (target - start))
            time.sleep(duration / (steps - 1))

    def open(self, duration: float = 2.0) -> None:
        """양쪽 그리퍼를 천천히 완전히 연다."""
        self.move_positions([0.0, 0.0], duration=duration)

    def close(self, target: float = 0.35, torque: float = 0.20, duration: float = 2.0) -> None:
        """양쪽 그리퍼를 지정한 토크와 부분 닫힘 위치까지 천천히 닫는다."""
        self.set_position_torque(torque)
        self.move_positions([target, target], duration=duration)
