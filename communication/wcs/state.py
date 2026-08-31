"""SDK RobotState와 데모 작업 상태를 WCS 전송용 형태로 변환하고 보관한다."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

WORK_CYCLES = ("IDLE", "WORKING", "DONE", "ERROR", "UNKNOWN")
PARTS = ("mobility", "torso", "right_arm", "left_arm", "head")


@dataclass(frozen=True)
class RobotSnapshot:
    """WCS payload 생성에 필요한 최신 로봇 상태."""

    received_at: float
    is_ready: bool
    emo_pressed: bool
    power_on: bool
    servo_on: bool
    battery_percent: float | None
    battery_voltage: float | None
    battery_current: float | None
    position: tuple[float, ...]
    joint_names: tuple[str, ...]
    idx: dict[str, tuple[int, ...]]
    odometry: dict[str, float | None] | None
    system: dict[str, float | None] | None


class WcsStateStore:
    """RobotState와 work cycle을 하나의 lock으로 안전하게 보관한다."""

    def __init__(self, robot_model: Any) -> None:
        if robot_model is None:
            raise ValueError("robot_model이 필요합니다.")

        self._lock = threading.Lock()
        self._latest_robot: RobotSnapshot | None = None
        self._work_cycle = "UNKNOWN"
        self._error_message: str | None = None

        # 모델 정보는 실행 중 바뀌지 않으므로 50Hz callback마다 다시 읽지 않는다.
        self._joint_names = tuple(str(name) for name in _attribute_or_empty(robot_model, "robot_joint_names"))
        self._idx = {
            part: tuple(int(index) for index in _attribute_or_empty(robot_model, f"{part}_idx"))
            for part in PARTS
        }

    def update_robot_state(self, state: Any) -> None:
        """SDK RobotState를 변환한 뒤 최신 snapshot으로 교체한다."""
        snapshot = self._to_snapshot(state)
        with self._lock:
            self._latest_robot = snapshot

    def latest_robot(self) -> RobotSnapshot | None:
        with self._lock:
            return self._latest_robot

    def set_work_state(self, cycle: str, error_message: str | None = None) -> None:
        """작업 상태를 갱신하며 ERROR가 아니면 이전 오류 메시지를 지운다."""
        normalized = cycle.strip().upper()
        if normalized not in WORK_CYCLES:
            log.warning("알 수 없는 work cycle %r -> UNKNOWN으로 대체", cycle)
            normalized = "UNKNOWN"

        message = (error_message or "").strip() or None
        if normalized != "ERROR":
            message = None

        with self._lock:
            self._work_cycle = normalized
            self._error_message = message

    def get_work_state(self) -> tuple[str, str | None]:
        with self._lock:
            return self._work_cycle, self._error_message

    def _to_snapshot(self, state: Any) -> RobotSnapshot:
        battery = getattr(state, "battery_state", None)
        emo_states = _attribute_or_empty(state, "emo_states")
        joint_states = _attribute_or_empty(state, "joint_states")
        power_states = _attribute_or_empty(state, "power_states")

        raw_odometry = getattr(state, "odometry", None)
        odometry = _se2_to_dict(raw_odometry) if raw_odometry is not None else None

        system_stat = getattr(state, "system_stat", None)
        system = None
        if system_stat is not None:
            system = {
                "cpu_usage": _num(getattr(system_stat, "cpu_usage", None)),
                "memory_usage": _num(getattr(system_stat, "memory_usage", None)),
                "uptime": _num(getattr(system_stat, "uptime", None)),
            }

        return RobotSnapshot(
            received_at=time.monotonic(),
            is_ready=_all_true(getattr(state, "is_ready", False)),
            emo_pressed=any(_enum_name(getattr(item, "state", None)) == "Pressed" for item in emo_states),
            power_on=(
                all(_enum_name(getattr(item, "state", None)) == "PowerOn" for item in power_states)
                if power_states
                else False
            ),
            servo_on=(
                all(bool(getattr(item, "power_on", False)) for item in joint_states)
                if joint_states
                else False
            ),
            battery_percent=_num(getattr(battery, "level_percent", None)),
            battery_voltage=_num(getattr(battery, "voltage", None)),
            battery_current=_num(getattr(battery, "current", None)),
            position=tuple(float(value) for value in _attribute_or_empty(state, "position")),
            joint_names=self._joint_names,
            idx=dict(self._idx),
            odometry=odometry,
            system=system,
        )


def _attribute_or_empty(value: Any, name: str) -> tuple[Any, ...]:
    attribute = getattr(value, name, None)
    if attribute is None:
        return ()
    try:
        return tuple(attribute)
    except TypeError:
        return (attribute,)


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", ""))


def _all_true(value: Any) -> bool:
    """스칼라 bool과 SDK의 관절별 bool 배열을 모두 처리한다."""
    if isinstance(value, bool):
        return value

    try:
        sequence = list(value)
    except TypeError:
        return bool(value)

    return bool(sequence) and all(bool(item) for item in sequence)


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def _se2_to_dict(raw: Any) -> dict[str, float | None] | None:
    """SDK odometry의 3x3 SE(2) 행렬과 호환 타입을 x/y/rz로 변환한다."""
    if getattr(raw, "shape", None) == (3, 3):
        try:
            x = _num(raw[0][2])
            y = _num(raw[1][2])
            cos_yaw = _num(raw[0][0])
            sin_yaw = _num(raw[1][0])
        except (TypeError, ValueError, IndexError):
            return None
        if None in (x, y, cos_yaw, sin_yaw):
            return None
        return {"x": x, "y": y, "rz": math.atan2(sin_yaw, cos_yaw)}

    for attributes in (("x", "y", "theta"), ("x", "y", "rz"), ("x", "y", "angle")):
        if all(hasattr(raw, attribute) for attribute in attributes):
            return {
                "x": _num(getattr(raw, attributes[0])),
                "y": _num(getattr(raw, attributes[1])),
                "rz": _num(getattr(raw, attributes[2])),
            }

    try:
        sequence = list(raw)
    except TypeError:
        return None

    if len(sequence) < 3:
        return None

    return {"x": _num(sequence[0]), "y": _num(sequence[1]), "rz": _num(sequence[2])}
