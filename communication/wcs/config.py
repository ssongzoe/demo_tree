"""SFA WCS 통신 설정. 모든 값은 환경변수로 덮어쓸 수 있다."""

from __future__ import annotations

import os


def _positive_float(key: str, default: float) -> float:
    raw = os.getenv(key, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{key}는 숫자여야 합니다: {raw!r}") from error

    if value <= 0:
        raise ValueError(f"{key}는 0보다 커야 합니다: {value}")
    return value


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key, str(default)).strip().lower()
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"{key}는 true/false 값이어야 합니다: {raw!r}")


# SFA 개발서버
WCS_BASE_URL = os.getenv("WCS_BASE_URL", "http://210.101.65.119:5224")
WCS_HEALTH_PATH = os.getenv("WCS_HEALTH_PATH", "/health")
WCS_STATUS_PATH = os.getenv("WCS_STATUS_PATH", "/api/v1/rb/rby1/status")
HTTP_TIMEOUT_SEC = _positive_float("HTTP_TIMEOUT_SEC", 5.0)

# 전송 대상과 heartbeat 주기
ROBOT_SERIAL = os.getenv("ROBOT_SERIAL", "RBY1-001")
UPLOAD_RATE_HZ = _positive_float("UPLOAD_RATE_HZ", 1.0)

# 실제 POST 없이 payload만 로그로 확인할 때 사용한다.
DRY_RUN = _bool("DRY_RUN", True) #False