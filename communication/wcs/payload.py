"""RobotSnapshot과 작업 상태를 SFA WCS status JSON으로 변환한다."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from .state import PARTS, WORK_CYCLES, RobotSnapshot

STALE_AFTER_SEC = 2.0
ERROR_MESSAGE_MAX_LEN = 200


def build_status_payload(
    serial: str,
    snapshot: RobotSnapshot,
    *,
    work_cycle: str,
    error_message: str | None,
) -> dict[str, Any]:
    """최신 로봇 상태와 작업 상태로 WCS 전송 payload 한 건을 만든다."""
    cycle = _normalize_work_cycle(work_cycle)
    message = _normalize_error_message(cycle, error_message)

    encoder = {
        part: [
            round(snapshot.position[index], 5)
            for index in snapshot.idx.get(part, ())
            if 0 <= index < len(snapshot.position)
        ]
        for part in PARTS
    }

    return {
        "robotSerial": serial,
        "robotType": "RBY1",
        "robot_state": {
            # WCS AMR status 규격에 맞춰 bool이 아닌 문자열 "true"/"false"를 사용한다.
            "emo": _bool_string(snapshot.emo_pressed),
            "power": _bool_string(snapshot.power_on),
            "servo": _bool_string(snapshot.servo_on),
            "control_ready": _bool_string(snapshot.is_ready),
            "work_cycle": cycle,
            "error_message": message,
        },
        "power": {
            "bat_percent": snapshot.battery_percent,
            "bat_voltage": snapshot.battery_voltage,
            "bat_current": snapshot.battery_current,
        },
        "encoder": encoder,
        "pose": dict(snapshot.odometry) if snapshot.odometry is not None else None,
        "system": dict(snapshot.system) if snapshot.system is not None else None,
        "isStale": (time.monotonic() - snapshot.received_at) > STALE_AFTER_SEC,
        "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    }


def _normalize_work_cycle(work_cycle: str) -> str:
    cycle = work_cycle.strip().upper()
    return cycle if cycle in WORK_CYCLES else "UNKNOWN"


def _normalize_error_message(work_cycle: str, error_message: str | None) -> str | None:
    if work_cycle != "ERROR":
        return None

    message = (error_message or "").strip()
    return message[:ERROR_MESSAGE_MAX_LEN] or None


def _bool_string(value: bool) -> str:
    return "true" if value else "false"