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


def _port(key: str, default: int) -> int:
    raw = os.getenv(key, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{key}는 정수여야 합니다: {raw!r}") from error

    if not 1 <= value <= 65535:
        raise ValueError(f"{key}는 1~65535 사이여야 합니다: {value}")
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
# 반송 완료/실패 보고 (AMR FRS→WCS transport-events callback 규격 준용)
WCS_TRANSPORT_EVENT_PATH = os.getenv("WCS_TRANSPORT_EVENT_PATH", "/api/v1/rb/transport-events")
HTTP_TIMEOUT_SEC = _positive_float("HTTP_TIMEOUT_SEC", 5.0)

# 로봇 측 반송 오더 수신 서버 (WCS → RBY1). AMR FRS의 POST /api/v1/wcs/transport-orders 규격 준용.
ROBOT_ORDER_BIND = os.getenv("ROBOT_ORDER_BIND", "0.0.0.0")
ROBOT_ORDER_PORT = _port("ROBOT_ORDER_PORT", 5225)
ROBOT_ORDER_PATH = os.getenv("ROBOT_ORDER_PATH", "/api/v1/wcs/transport-orders")

# 전송 대상과 heartbeat 주기
ROBOT_SERIAL = os.getenv("ROBOT_SERIAL", "RBY1-001")
UPLOAD_RATE_HZ = _positive_float("UPLOAD_RATE_HZ", 1.0)

# 실제 POST 없이 payload만 로그로 확인할 때 사용한다.
DRY_RUN = _bool("DRY_RUN", True) #False
