"""내부 테스트용 가짜 SFA WCS 서버 (표준 라이브러리만 사용).

    python tools/fake_wcs/server.py                                   # http://0.0.0.0:5224, 수동 발행
    FAKE_WCS_AUTO_DISPATCH=1 python tools/fake_wcs/server.py          # 자동 반복 발행
    FAKE_WCS_ROBOT_URL=http://192.168.30.10:5225 python tools/fake_wcs/server.py
    FAKE_WCS_COOLDOWN_SEC=15 FAKE_WCS_PORT=5224 python tools/fake_wcs/server.py

AMR 쪽 SFA-WCS 규격(v07.2)의 Transport Order / transport-events를 RBY1에 맞게 축소해 흉내 낸다.
WCS가 로봇 측 오더 서버에 오더를 POST(push)하고, 로봇이 완료/실패를 콜백한다.

기본은 **수동 발행**: 대시보드 "오더 발행" 카드에서 [오더 발행]을 눌러야 오더가 나간다.
자동 발행(FAKE_WCS_AUTO_DISPATCH=1 또는 대시보드 토글)이면 READY가 될 때마다 자동으로 발행한다.

    READY ──(오더 POST → 로봇 ACCEPTED)──> RUNNING
    RUNNING ──(COMPLETED 콜백)──> COOLDOWN(15초) ──(만료)──> READY (다음 오더 발행)
    RUNNING ──(FAILED 콜백)──> HALTED ──(대시보드 [재개] / POST /api/test/resume)──> READY
    RUNNING ──(대시보드 [취소] / POST /api/test/cancel → 로봇에 취소 POST)──> CANCEL_REQUESTED
    CANCEL_REQUESTED ──(CANCELED 콜백)──> COOLDOWN ──> READY
    로봇에 연결이 안 되면 READY에서 2초마다 재시도한다.

로봇 → WCS (실 WCS 호환):
    GET  /health                                    -> "healthy"
    GET  /api/v1/rb/stations/{stationId}/readiness?operation=LOAD|UNLOAD&wcsOrderId=
                                                    -> 200 {stationId, operation, status: READY|NOT_READY, ready, reasonCode, updatedAt}
                                                       (v07.4 9장 Application PIO. 기본 READY, 대시보드/POST /api/test/readiness로 변경)
    POST /api/v1/rb/rby1/status                     -> 201 {"accepted": true, "recordId": ...}
    POST /api/v1/rb/transport-events                -> 200 {"accepted": true, "eventId": ..., "receivedAt": ...}
    GET  /api/v1/rb/rby1/status/{serial}/latest | /history?limit=N
WCS → 로봇 (이 서버가 호출):
    POST {FAKE_WCS_ROBOT_URL}/api/v1/wcs/transport-orders
    POST {FAKE_WCS_ROBOT_URL}/api/v1/wcs/transport-orders/{wcsOrderId}/cancel
테스트 전용:
    POST /api/test/order   {wcsOrderId?, carrierId?, fromStationId?, toStationId?, priority?}  수동 발행
    POST /api/test/auto    {"enabled": true|false}                                             자동 발행 토글
    POST /api/test/readiness {stationId, operation, status: READY|NOT_READY, reasonCode?}      PIO readiness 설정
    POST /api/test/resume, POST /api/test/cancel, GET /api/test/state, GET /  (대시보드)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

PORT = int(os.getenv("FAKE_WCS_PORT", "5226"))
BIND = os.getenv("FAKE_WCS_BIND", "0.0.0.0")
COOLDOWN_SEC = float(os.getenv("FAKE_WCS_COOLDOWN_SEC", "15"))
ROBOT_URL = os.getenv("FAKE_WCS_ROBOT_URL", "http://127.0.0.1:3000").rstrip("/")
ROBOT_ORDER_PATH = os.getenv("FAKE_WCS_ROBOT_ORDER_PATH", "/api/v1/wcs/transport-orders")
FROM_STATION = os.getenv("FAKE_WCS_FROM_STATION", "RACK01_PORT01")
TO_STATION = os.getenv("FAKE_WCS_TO_STATION", "CV02_IN")
CARRIER_PREFIX = os.getenv("FAKE_WCS_CARRIER_PREFIX", "TOTE")
AUTO_DISPATCH = os.getenv("FAKE_WCS_AUTO_DISPATCH", "0").strip().lower() in ("1", "true", "yes", "y", "on")
ORDER_FIELDS = ("wcsOrderId", "carrierId", "fromStationId", "toStationId", "priority")
ORDER_RETRY_SEC = 2.0
HTTP_TIMEOUT = 5.0
HISTORY_MAX = 1000
LOG_MAX = 100
BUSY = -1  # _dispatch: 다른 발행이 진행 중이라 건너뜀

STATUS_PREFIX = "/api/v1/rb/rby1/status"
EVENT_PATH = "/api/v1/rb/transport-events"
STATIONS_PREFIX = "/api/v1/rb/stations"
READINESS_OPERATIONS = ("LOAD", "UNLOAD")
ARRIVAL_EVENTS = ("ARRIVED_AT_FROM", "ARRIVED_AT_TO")

log = logging.getLogger("fake-wcs")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _str_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


class WcsSimulator:
    """수신 status 저장 + 반송 오더 발행 상태 머신. 모든 상태는 lock 하나로 보호한다."""

    def __init__(self, cooldown_sec: float, auto_dispatch: bool) -> None:
        self._lock = threading.Lock()
        self._cooldown_sec = cooldown_sec
        self._auto = auto_dispatch
        self._state = "READY"
        self._order_seq = 0
        self._current_order: dict[str, Any] | None = None
        self._cooldown_until: float | None = None
        self._last_error: dict[str, Any] | None = None
        self._robot_reachable: bool | None = None
        self._robot_error_logged = False
        self._dispatching = False
        self._history: dict[str, deque[dict[str, Any]]] = {}
        self._order_log: deque[dict[str, Any]] = deque(maxlen=LOG_MAX)
        self._event_log: deque[dict[str, Any]] = deque(maxlen=LOG_MAX)
        # 통신 로그: 보낸 오더/취소 요청과 로봇 응답, 로봇이 보낸 이벤트와 우리 응답 (원본 그대로)
        self._comm_log: deque[dict[str, Any]] = deque(maxlen=LOG_MAX)
        # PIO readiness (v07.4 9장). key "stationId:OPERATION". 항목이 없으면 READY.
        self._readiness: dict[str, dict[str, Any]] = {}
        # 로봇이 지금 무엇을 묻고 있는지 (대시보드 표시용). key 동일.
        self._readiness_polls: dict[str, dict[str, Any]] = {}
        self._seen_event_ids: dict[str, str] = {}
        self._received = 0

    # ── 디스패처 스레드: 오더 발행 ────────────────────────────
    def run_dispatcher(self, stop: threading.Event) -> None:
        while not stop.is_set():
            with self._lock:
                if self._state == "COOLDOWN" and self._cooldown_until is not None \
                        and time.monotonic() >= self._cooldown_until:
                    self._cooldown_until = None
                    self._transition("READY", "대기 시간 만료")
                should_dispatch = self._state == "READY" and self._auto

            if should_dispatch:
                self._dispatch_order()
            stop.wait(ORDER_RETRY_SEC if not should_dispatch or self._robot_reachable is False else 0.5)

    def set_auto(self, enabled: bool) -> bool:
        """자동 발행 토글 (대시보드). 켜면 디스패처가 다음 루프에서 READY면 바로 발행한다."""
        with self._lock:
            if self._auto != enabled:
                self._auto = enabled
                log.info("발행 모드: %s", "자동" if enabled else "수동")
            return self._auto

    def dispatch_manual(self, fields: dict[str, Any]) -> tuple[bool, str, int]:
        """대시보드 폼/curl로 오더 1건을 즉시 발행한다. (ok, message, http_code)"""
        with self._lock:
            if self._state == "COOLDOWN":
                self._cooldown_until = None
                self._transition("READY", "수동 발행으로 대기 종료")
            if self._state != "READY":
                return False, f"발행 불가 (state={self._state}) — 진행 중 오더가 끝나거나 [재개] 후 다시 시도", 409
            order = self._build_order(self._order_seq + 1)

        for key in ORDER_FIELDS:
            value = fields.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            if key == "priority":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    return False, f"priority는 정수여야 합니다: {value!r}", 400
            else:
                value = str(value).strip()
            order[key] = value

        code, body = self._dispatch(order)
        if code == BUSY:
            return False, body["error"], 409
        if code is None:
            return False, f"로봇 오더 서버 연결 실패: {body.get('error')}", 502
        if self._state == "RUNNING" and self._current_order and \
                self._current_order["wcsOrderId"] == order["wcsOrderId"]:
            return True, f"발행 완료 {order['wcsOrderId']} ({order['fromStationId']} → {order['toStationId']}) HTTP {code}", 200
        return False, f"오더 거절 HTTP {code}: {body}", 409

    def _record_comm(self, direction: str, kind: str, url: str, request: Any,
                     code: int | None, response: Any, order_id: str | None = None) -> None:
        """대시보드 통신 로그 1건 (lock 밖에서 호출해도 되도록 자체 lock)."""
        entry = {"at": _now_iso(), "direction": direction, "kind": kind, "url": url,
                 "wcsOrderId": order_id, "request": request, "httpCode": code, "response": response}
        with self._lock:
            self._comm_log.appendleft(entry)

    # ── PIO readiness (v07.4 9장) ───────────────────────────────
    @staticmethod
    def _readiness_key(station_id: str, operation: str) -> str:
        return f"{station_id}:{operation.upper()}"

    def set_readiness(self, station_id: str, operation: str, status: str,
                      reason_code: str | None) -> dict[str, Any]:
        """대시보드/curl로 설비 준비 상태를 바꾼다."""
        entry = {"stationId": station_id, "operation": operation.upper(), "status": status.upper(),
                 "reasonCode": (reason_code or None) if status.upper() != "READY" else None,
                 "updatedAt": _now_iso()}
        with self._lock:
            self._readiness[self._readiness_key(station_id, operation)] = entry
        log.info("readiness 설정: %s %s -> %s (reasonCode=%s)", station_id, operation.upper(),
                 entry["status"], entry["reasonCode"])
        return dict(entry)

    def get_readiness(self, station_id: str, operation: str, wcs_order_id: str | None) -> dict[str, Any]:
        """로봇의 readiness 조회 응답. 통신 로그에는 첫 조회와 status가 바뀐 조회만 남긴다."""
        operation = operation.upper()
        key = self._readiness_key(station_id, operation)
        with self._lock:
            entry = self._readiness.get(key)
            body = {"stationId": station_id, "operation": operation,
                    "status": entry["status"] if entry else "READY",
                    "ready": (entry["status"] == "READY") if entry else True,
                    "reasonCode": entry["reasonCode"] if entry else None,
                    "updatedAt": entry["updatedAt"] if entry else _now_iso()}
            poll = self._readiness_polls.get(key)
            first_or_changed = poll is None or poll.get("lastStatus") != body["status"] \
                or poll.get("wcsOrderId") != wcs_order_id
            if first_or_changed:
                poll = {"count": 0, "firstAt": _now_iso()}
                self._readiness_polls[key] = poll
            poll.update({"count": poll["count"] + 1, "lastAt": _now_iso(), "wcsOrderId": wcs_order_id,
                         "lastStatus": body["status"], "stationId": station_id, "operation": operation})
        if first_or_changed:
            log.info("readiness 조회: %s %s (order=%s) -> %s%s", station_id, operation, wcs_order_id, body["status"],
                     "" if body["ready"] else f" (reasonCode={body['reasonCode']}) — 로봇이 재확인 중")
            self._record_comm("로봇→WCS", f"readiness {operation}",
                              f"GET {STATIONS_PREFIX}/{station_id}/readiness?operation={operation}&wcsOrderId={wcs_order_id}",
                              {"stationId": station_id, "operation": operation, "wcsOrderId": wcs_order_id},
                              200, body, wcs_order_id)
        return body

    def cancel_current(self) -> tuple[bool, str]:
        """진행 중 오더의 취소를 로봇에 요청한다 (v07.3 4.8). 대시보드 [취소] 버튼용."""
        with self._lock:
            if self._state != "RUNNING" or self._current_order is None:
                return False, f"취소할 진행 중 오더가 없습니다 (state={self._state})"
            order_id = self._current_order["wcsOrderId"]

        payload = {"reasonCode": "OPERATOR_REQUEST", "reason": "가짜 WCS 대시보드 수동 취소",
                   "requestedAt": _now_iso()}
        code, body = self._post_robot_path(f"/{order_id}/cancel", payload, f"CANCEL-{order_id}")
        self._record_comm("WCS→로봇", "취소 요청", f"POST {ROBOT_URL}{ROBOT_ORDER_PATH}/{order_id}/cancel",
                          payload, code, body, order_id)
        with self._lock:
            if code in (200, 202) and isinstance(body, dict):
                status = body.get("orderStatus", "CANCEL_REQUESTED")
                if self._current_order and self._current_order["wcsOrderId"] == order_id:
                    self._current_order["orderStatus"] = status
                self._transition("CANCEL_REQUESTED", f"취소 요청 {order_id} -> 로봇 {status} (HTTP {code})")
                return True, f"취소 요청 접수 (HTTP {code}, {status})"
            message = f"취소 요청 실패 HTTP {code}: {body}"
            log.warning("%s", message)
            return False, message

    @staticmethod
    def _build_order(seq: int) -> dict[str, Any]:
        return {
            "wcsOrderId": f"WCS-{datetime.now().strftime('%Y%m%d')}-{seq:06d}",
            "carrierId": f"{CARRIER_PREFIX}-{seq:06d}",
            "fromStationId": FROM_STATION,
            "toStationId": TO_STATION,
            "priority": 5,
            "timestamp": _now_iso(),
        }

    def _dispatch_order(self) -> None:
        """자동 발행: 기본값 오더를 만들어 보낸다."""
        with self._lock:
            order = self._build_order(self._order_seq + 1)
        self._dispatch(order)

    def _dispatch(self, order: dict[str, Any]) -> tuple[int | None, Any]:
        """오더를 로봇에 POST하고 결과에 따라 상태 전이. (code, body) 반환. 동시 발행은 BUSY로 거른다."""
        with self._lock:
            if self._state != "READY" or self._dispatching:
                return BUSY, {"error": f"발행 불가 (state={self._state}, dispatching={self._dispatching})"}
            self._dispatching = True
        try:
            code, body = self._post_robot(order)
        finally:
            with self._lock:
                self._dispatching = False
        self._record_comm("WCS→로봇", "오더 발행", f"POST {ROBOT_URL}{ROBOT_ORDER_PATH}",
                          order, code, body, order["wcsOrderId"])
        with self._lock:
            seq = self._order_seq + 1
            if code is None:
                if not self._robot_error_logged:
                    log.warning("로봇 오더 서버에 연결할 수 없습니다 (%s%s): %s — %.0f초마다 재시도",
                                ROBOT_URL, ROBOT_ORDER_PATH, body.get("error"), ORDER_RETRY_SEC)
                    self._robot_error_logged = True
                self._robot_reachable = False
                return code, body

            self._robot_reachable = True
            self._robot_error_logged = False
            if code in (200, 201) and isinstance(body, dict) and body.get("orderStatus") == "ACCEPTED":
                self._order_seq = seq
                self._current_order = {**order, "orderStatus": "ACCEPTED", "acceptedAt": _now_iso()}
                self._order_log.appendleft(dict(self._current_order))
                self._transition("RUNNING", f"오더 발행 {order['wcsOrderId']} -> 로봇 ACCEPTED (HTTP {code})")
            else:
                self._last_error = {"wcsOrderId": order["wcsOrderId"], "message": f"오더 거절 HTTP {code}: {body}",
                                    "receivedAt": _now_iso()}
                self._transition("HALTED", f"오더 발행 실패 HTTP {code}: {body}")
            return code, body

    @staticmethod
    def _post_robot(order: dict[str, Any]) -> tuple[int | None, Any]:
        return WcsSimulator._post_robot_path("", order, order["wcsOrderId"])

    @staticmethod
    def _post_robot_path(subpath: str, payload: dict[str, Any],
                         correlation_id: str) -> tuple[int | None, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            ROBOT_URL + ROBOT_ORDER_PATH + subpath, data=data, method="POST",
            headers={"Content-Type": "application/json; charset=utf-8",
                     "X-Correlation-Id": correlation_id})
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                return response.status, json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.loads(error.read().decode("utf-8") or "{}")
            except ValueError:
                return error.code, {"error": str(error)}
        except (urllib.error.URLError, OSError, ValueError) as error:
            return None, {"error": str(error)}

    # ── 로봇 → WCS 콜백 ────────────────────────────────────────
    def on_transport_event(self, event: dict[str, Any]) -> dict[str, Any]:
        event_id = str(event.get("eventId") or uuid.uuid4().hex)
        record = {**event, "eventId": event_id, "receivedAt": _now_iso()}

        with self._lock:
            if event_id in self._seen_event_ids:
                log.info("중복 이벤트 (멱등 처리): %s", event_id)
                reply = {"accepted": True, "eventId": event_id, "receivedAt": self._seen_event_ids[event_id]}
                self._comm_log.appendleft({"at": _now_iso(), "direction": "로봇→WCS", "kind": "이벤트(중복)",
                                           "url": f"POST {EVENT_PATH}", "wcsOrderId": event.get("wcsOrderId"),
                                           "request": event, "httpCode": 200, "response": reply})
                return reply
            self._seen_event_ids[event_id] = record["receivedAt"]
            self._event_log.appendleft(record)

            order_id = str(event.get("wcsOrderId") or "")
            event_type = str(event.get("eventType") or "").upper()
            current_id = self._current_order["wcsOrderId"] if self._current_order else None
            log.info("transport-event 수신: %s %s result=%s message=%r%s", event_type, order_id,
                     event.get("result"), event.get("message"),
                     f" nodeId={event.get('nodeId')} (도착 보고 — 상태 전이 없음)" if event_type in ARRIVAL_EVENTS else "")

            if order_id != current_id:
                log.warning("현재 오더(%s)가 아닌 이벤트: %s — 상태 전이 없음", current_id, order_id)
            elif event_type == "COMPLETED":
                self._current_order["orderStatus"] = "COMPLETED"
                self._cooldown_until = time.monotonic() + self._cooldown_sec
                self._transition("COOLDOWN", f"{order_id} 완료, {self._cooldown_sec:.0f}초 대기")
            elif event_type == "CANCELED":
                self._current_order["orderStatus"] = "CANCELED"
                self._cooldown_until = time.monotonic() + self._cooldown_sec
                self._transition("COOLDOWN", f"{order_id} 취소 완료, {self._cooldown_sec:.0f}초 대기")
            elif event_type == "FAILED":
                self._current_order["orderStatus"] = "FAILED"
                self._last_error = {"wcsOrderId": order_id, "message": event.get("message"),
                                    "receivedAt": record["receivedAt"], "robotSerial": event.get("robotSerial")}
                self._transition("HALTED", f"{order_id} 실패: {event.get('message')!r}")
            else:
                self._current_order["orderStatus"] = event_type or self._current_order["orderStatus"]

        reply = {"accepted": True, "eventId": event_id, "receivedAt": record["receivedAt"]}
        self._record_comm("로봇→WCS", f"이벤트 {event_type or '?'}", f"POST {EVENT_PATH}", event, 200, reply,
                          order_id or None)
        return reply

    # ── status 수신 ─────────────────────────────────────────────
    def on_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = _flatten(payload)
        with self._lock:
            self._received += 1
            self._history.setdefault(record["robotSerial"], deque(maxlen=HISTORY_MAX)).appendleft(record)
        if self._received % 10 == 0:
            log.info("status 수신 %d건 (최근: serial=%s cycle=%s)", self._received,
                     record["robotSerial"], record["workCycle"])
        return record

    def resume(self) -> None:
        with self._lock:
            if self._state == "HALTED":
                self._last_error = None
                self._transition("READY", "수동 재개")

    def _transition(self, new_state: str, reason: str) -> None:
        log.info("[%s -> %s] %s", self._state, new_state, reason)
        self._state = new_state

    # ── 조회 ───────────────────────────────────────────────────
    def latest(self, serial: str) -> dict[str, Any] | None:
        with self._lock:
            records = self._history.get(serial)
            return dict(records[0]) if records else None

    def history(self, serial: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in list(self._history.get(serial, ()))[:limit]]

    def dashboard_state(self) -> dict[str, Any]:
        with self._lock:
            remaining = None
            if self._state == "COOLDOWN" and self._cooldown_until is not None:
                remaining = max(0.0, self._cooldown_until - time.monotonic())
            return {
                "serverState": self._state,
                "autoDispatch": self._auto,
                "fromStation": FROM_STATION,
                "toStation": TO_STATION,
                "carrierPrefix": CARRIER_PREFIX,
                "nextSeq": self._order_seq + 1,
                "cooldownSec": self._cooldown_sec,
                "cooldownRemaining": remaining,
                "robotUrl": ROBOT_URL + ROBOT_ORDER_PATH,
                "robotReachable": self._robot_reachable,
                "currentOrder": dict(self._current_order) if self._current_order else None,
                "lastError": dict(self._last_error) if self._last_error else None,
                "orderLog": list(self._order_log),
                "eventLog": list(self._event_log),
                "commLog": list(self._comm_log),
                "readiness": list(self._readiness.values()),
                "readinessPolls": list(self._readiness_polls.values()),
                "received": self._received,
                "latest": {serial: dict(records[0]) for serial, records in self._history.items() if records},
                "serverTime": _now_iso(),
            }


def _flatten(payload: dict[str, Any]) -> dict[str, Any]:
    """실 WCS가 저장하는 평탄화 레코드 형태(monitor.py 기준) + errorMessage 컬럼."""
    robot_state = payload.get("robot_state") or {}
    power = payload.get("power") or {}
    pose = payload.get("pose") or {}
    system = payload.get("system") or {}
    return {
        "recordId": uuid.uuid4().hex,
        "robotSerial": str(payload.get("robotSerial") or "UNKNOWN"),
        "robotType": str(payload.get("robotType") or ""),
        "receivedAt": _now_iso(),
        "sourceTime": payload.get("time"),
        "workCycle": str(robot_state.get("work_cycle") or "UNKNOWN").upper(),
        "errorMessage": robot_state.get("error_message"),
        "sourceIsStale": bool(payload.get("isStale", False)),
        "emergencyStop": _str_bool(robot_state.get("emo")),
        "mainPowerOn": _str_bool(robot_state.get("power")),
        "servoOn": _str_bool(robot_state.get("servo")),
        "controlReady": _str_bool(robot_state.get("control_ready")),
        "batteryPercent": power.get("bat_percent"),
        "batteryVoltage": power.get("bat_voltage"),
        "batteryCurrent": power.get("bat_current"),
        "poseX": pose.get("x"),
        "poseY": pose.get("y"),
        "poseRz": pose.get("rz"),
        "cpuUsage": system.get("cpu_usage"),
        "memoryUsage": system.get("memory_usage"),
        "uptimeSeconds": system.get("uptime"),
        "payloadJson": json.dumps(payload, ensure_ascii=False),
    }


SIM = WcsSimulator(COOLDOWN_SEC, AUTO_DISPATCH)


PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>가짜 WCS — RBY1 테스트 서버</title>
<style>
  :root { --bg:#0f1419; --panel:#1a2128; --line:#2b3540; --txt:#e6edf3; --dim:#8b98a5;
          --ok:#3fb950; --bad:#f85149; --warn:#d29922; --info:#58a6ff; }
  * { box-sizing:border-box; }
  body { margin:0; padding:18px; background:var(--bg); color:var(--txt);
         font:14px/1.5 -apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif; }
  h1 { font-size:17px; margin:0 0 2px; }
  .sub { color:var(--dim); font-size:12px; margin-bottom:14px; }
  .grid { display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
  .card h2 { font-size:12px; color:var(--dim); margin:0 0 10px; font-weight:600;
             letter-spacing:.04em; text-transform:uppercase; }
  .tiles { display:flex; gap:8px; flex-wrap:wrap; }
  .tile { flex:1 1 90px; text-align:center; padding:10px 6px; border-radius:8px;
          background:#000; border:1px solid var(--line); }
  .tile .k { font-size:11px; color:var(--dim); }
  .tile .v { font-size:16px; font-weight:700; margin-top:3px; }
  .ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)} .dim{color:var(--dim)} .info{color:var(--info)}
  .row { display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid var(--line); gap:10px; }
  .row:last-child { border-bottom:0; }
  .row .k { color:var(--dim); white-space:nowrap; }
  .row .v { font-variant-numeric:tabular-nums; font-weight:600; text-align:right; word-break:break-all; }
  .bar { height:9px; background:#000; border-radius:5px; overflow:hidden; margin:8px 0 4px; }
  .bar > i { display:block; height:100%; background:var(--ok); transition:width .4s; }
  .banner { padding:10px 12px; border-radius:8px; margin-bottom:12px; font-weight:600; }
  .banner.err { background:#3d1418; border:1px solid var(--bad); color:#ffb4ae; }
  .banner.warn{ background:#3a2d10; border:1px solid var(--warn); color:#f0d48a; }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  th,td { text-align:right; padding:3px 6px; border-bottom:1px solid var(--line); font-size:12px; }
  th:first-child, td:first-child { text-align:left; }
  th { color:var(--dim); font-weight:600; }
  .grp { color:var(--info); font-weight:700; padding-top:8px; }
  .full { grid-column:1/-1; }
  button { background:var(--info); color:#000; border:0; border-radius:6px; padding:6px 14px;
           font-weight:700; cursor:pointer; }
  button:disabled { opacity:.35; cursor:not-allowed; }
  button.danger { background:var(--bad); color:#fff; }
  button.okbtn { background:var(--ok); color:#000; }
  .form .row { align-items:center; }
  .form input[type=text], .form input[type=number] { flex:1; min-width:0; background:#000; color:var(--txt);
           border:1px solid var(--line); border-radius:6px; padding:5px 8px; font:inherit; text-align:right; }
  .form input::placeholder { color:var(--dim); opacity:.7; }
  .btns { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
  .toggle { display:flex; align-items:center; gap:6px; margin-bottom:8px; cursor:pointer; }
  .toggle input { width:16px; height:16px; accent-color:var(--ok); }
  #orderMsg { min-height:1.5em; margin-top:8px; font-size:12px; word-break:break-all; }
  .rd { display:grid; grid-template-columns:auto 1fr 1fr; gap:6px 10px; align-items:center; font-size:12px; }
  .rd .st { font-weight:700; font-variant-numeric:tabular-nums; }
  .rd .op { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
  .rd .op button { padding:3px 8px; font-size:11px; }
  .rd .op button.off { background:#000; color:var(--dim); border:1px solid var(--line); }
  .rd .poll { font-size:11px; color:var(--warn); }
  .comm { display:flex; flex-direction:column; gap:10px; max-height:640px; overflow:auto; }
  .comm .entry { border:1px solid var(--line); border-radius:8px; padding:8px 10px; }
  .comm .head { display:flex; gap:10px; flex-wrap:wrap; align-items:center; font-size:12px;
                font-variant-numeric:tabular-nums; }
  .comm .head .dir { font-weight:700; min-width:70px; }
  .comm .head .kind { min-width:90px; }
  .comm .head .code { min-width:60px; font-weight:700; }
  .comm .head .url { color:var(--dim); flex:1; word-break:break-all; }
  .comm .pair { display:grid; gap:8px; grid-template-columns:1fr 1fr; margin-top:6px; }
  .comm .pair pre { max-height:none; }
  .comm .pair .k { font-size:11px; color:var(--dim); margin-bottom:3px; }
  @media (max-width:800px) { .comm .pair { grid-template-columns:1fr; } }
  .big { font-size:26px; font-weight:800; }
  #dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--dim);
         margin-right:5px; vertical-align:middle; }
  pre { background:#000; padding:10px; border-radius:8px; font-size:11px; overflow:auto; max-height:260px; margin:0; }
</style></head><body>
<h1><span id="dot"></span>가짜 WCS — RBY1 테스트 서버</h1>
<div class="sub" id="meta">연결 중…</div>
<div id="banner"></div>
<div class="grid">
  <div class="card"><h2>서버 상태 / 현재 오더</h2><div id="server"></div></div>
  <div class="card form"><h2>오더 발행 (WCS → 로봇)</h2>
    <label class="toggle"><input type="checkbox" id="auto" onchange="toggleAuto(this)"> 자동 발행 (완료·취소 후 대기 시간 지나면 반복)</label>
    <div class="row"><span class="k">wcsOrderId</span><input type="text" id="f_wcsOrderId" placeholder="비우면 자동"></div>
    <div class="row"><span class="k">carrierId</span><input type="text" id="f_carrierId" placeholder="비우면 자동"></div>
    <div class="row"><span class="k">fromStationId</span><input type="text" id="f_fromStationId"></div>
    <div class="row"><span class="k">toStationId</span><input type="text" id="f_toStationId"></div>
    <div class="row"><span class="k">priority</span><input type="number" id="f_priority" value="5" min="0" step="1"></div>
    <div class="btns">
      <button id="btnSend" class="okbtn" onclick="sendOrder()" disabled>오더 발행</button>
      <button id="btnCancel" class="danger" onclick="cancelOrder()" disabled>현재 오더 취소</button>
      <button id="btnResume" onclick="resume()" disabled>재개</button>
    </div>
    <div id="orderMsg" class="dim"></div>
  </div>
  <div class="card form"><h2>PIO Readiness (v07.4 9장, 로봇 → WCS GET)</h2>
    <div class="dim" style="font-size:12px;margin-bottom:8px">로봇은 파지 전 LOAD(from), 배치 전 UNLOAD(to)를 묻고 READY일 때만 진행. NOT_READY면 2초마다 재확인, 최대 대기 초과 시 FAILED.</div>
    <div id="readiness"></div>
    <div class="row" style="border:0;margin-top:6px"><span class="k">stationId 추가</span><input type="text" id="r_station" placeholder="예: CV03_IN"><button onclick="addStation()" style="margin-left:8px">추가</button></div>
    <div id="readyMsg" class="dim" style="min-height:1.5em;margin-top:6px;font-size:12px"></div>
  </div>
  <div class="card"><h2>로봇 상태</h2><div class="tiles" id="state"></div></div>
  <div class="card"><h2>배터리</h2><div id="batt"></div></div>
  <div class="card"><h2>위치 (오도메트리)</h2><div id="pose"></div></div>
  <div class="card"><h2>제어 PC</h2><div id="sys"></div></div>
  <div class="card"><h2>오더 이력 (WCS → 로봇)</h2><div id="orderlog"></div></div>
  <div class="card"><h2>이벤트 이력 (로봇 → WCS)</h2><div id="eventlog"></div></div>
  <div class="card full"><h2>통신 로그 — 오더 / 취소 / 이벤트 원본 (요청 → 응답)</h2><div id="commlog"></div></div>
  <div class="card full"><h2>엔코더 — 관절 (rad / deg)</h2><div id="enc"></div></div>
  <div class="card full"><h2>마지막 수신 status payload (원본)</h2><pre id="raw"></pre></div>
</div>
<script>
const PARTS = {
  mobility:  ["모빌리티", ["wheel_fr 앞-우","wheel_fl 앞-좌","wheel_rr 뒤-우","wheel_rl 뒤-좌"]],
  torso:     ["토르소",   ["torso_0 발목롤","torso_1 발목피치","torso_2 무릎","torso_3 힙피치","torso_4 힙롤","torso_5 허리요"]],
  right_arm: ["오른팔",   ["어깨피치","어깨롤","어깨요","팔꿈치","손목요1","손목피치","손목요2"]],
  left_arm:  ["왼팔",     ["어깨피치","어깨롤","어깨요","팔꿈치","손목요1","손목피치","손목요2"]],
  head:      ["헤드",     ["head_0 팬","head_1 틸트"]],
};
const deg = r => (r * 180 / Math.PI);
const f = (v, n=2) => (v === null || v === undefined) ? "—" : Number(v).toFixed(n);
const ts = s => s ? new Date(s).toLocaleTimeString('ko-KR') : "—";
const STATE_CLS = {READY:"info", RUNNING:"ok", COOLDOWN:"warn", CANCEL_REQUESTED:"warn", HALTED:"bad"};
const boolTile = (k, on, invert=false) => {
  const good = invert ? !on : on;
  return `<div class="tile"><div class="k">${k}</div>
          <div class="v ${good?'ok':'bad'}">${on?'ON':'OFF'}</div></div>`;
};

let formInit = false;
const $ = id => document.getElementById(id);
const msg = (text, cls) => { const m = $("orderMsg"); m.textContent = text; m.className = cls; };

async function post(path, body) {
  const r = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"},
                               body: body === undefined ? null : JSON.stringify(body)});
  let b = {};
  try { b = await r.json(); } catch (e) { b = {ok:false, message:"HTTP " + r.status}; }
  return b;
}

async function resume() {
  await post("/api/test/resume");
  msg("재개 → READY", "info");
  tick();
}

async function cancelOrder() {
  $("btnCancel").disabled = true;
  const b = await post("/api/test/cancel");
  msg(b.message, b.ok ? "warn" : "bad");
  tick();
}

async function sendOrder() {
  const fields = {};
  for (const k of ["wcsOrderId","carrierId","fromStationId","toStationId","priority"]) {
    const v = $("f_" + k).value.trim();
    if (v !== "") fields[k] = v;
  }
  $("btnSend").disabled = true;
  msg("발행 중…", "dim");
  const b = await post("/api/test/order", fields);
  msg(b.message, b.ok ? "ok" : "bad");
  if (b.ok) { $("f_wcsOrderId").value = ""; $("f_carrierId").value = ""; }
  tick();
}

async function toggleAuto(el) {
  const b = await post("/api/test/auto", {enabled: el.checked});
  msg(b.autoDispatch ? "자동 발행 ON — READY가 되면 자동으로 발행합니다" : "자동 발행 OFF — [오더 발행]으로만 발행합니다", "info");
  tick();
}

let commKey = "";
function renderComm(list) {
  const key = list.length ? list[0].at + list.length : "";
  if (key === commKey) return;  // 새 항목 없으면 다시 그리지 않음 (스크롤 위치 유지)
  commKey = key;
  const el = $("commlog");
  if (!list.length) { el.innerHTML = `<div class="dim">아직 주고받은 오더/취소/이벤트 없음</div>`; return; }
  const esc = v => String(v).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
  const codeCls = c => c === null ? "bad" : c < 300 ? "ok" : c < 500 ? "warn" : "bad";
  el.innerHTML = `<div class="comm">` + list.map(e => {
    const fromWcs = e.direction.startsWith("WCS");
    const kindCls = /FAILED/.test(e.kind) ? "bad" : /CANCEL/.test(e.kind) ? "warn" : "";
    return `<div class="entry">
      <div class="head"><span class="dim">${ts(e.at)}</span><span class="dir ${fromWcs ? "info" : "ok"}">${esc(e.direction)}</span>
        <span class="kind ${kindCls}">${esc(e.kind)}</span><span>${esc(e.wcsOrderId ?? "—")}</span>
        <span class="code ${codeCls(e.httpCode)}">${e.httpCode === null ? "연결 실패" : "HTTP " + e.httpCode}</span>
        <span class="url">${esc(e.url)}</span></div>
      <div class="pair">
        <div><div class="k">요청 (${fromWcs ? "WCS가 보냄" : "로봇이 보냄"})</div><pre>${esc(JSON.stringify(e.request, null, 2))}</pre></div>
        <div><div class="k">응답 (${fromWcs ? "로봇 회신" : "WCS 회신"})</div><pre>${esc(JSON.stringify(e.response, null, 2))}</pre></div>
      </div></div>`;
  }).join("") + `</div>`;
}

const extraStations = new Set();
function addStation() {
  const v = $("r_station").value.trim();
  if (v) { extraStations.add(v); $("r_station").value = ""; commKey = ""; readyKey = ""; tick(); }
}
async function setReadiness(station, op, status) {
  const reason = status === "NOT_READY" ? (prompt(`${station} ${op} reasonCode (선택, 비워도 됨)`, "") || null) : null;
  const b = await post("/api/test/readiness", {stationId: station, operation: op, status, reasonCode: reason});
  const m = $("readyMsg"); m.textContent = b.ok ? `${station} ${op} → ${status}${reason ? " (" + reason + ")" : ""}` : b.message;
  m.className = b.ok ? (status === "READY" ? "ok" : "warn") : "bad";
  readyKey = ""; tick();
}
let readyKey = "";
function renderReadiness(s) {
  const map = {};
  for (const r of s.readiness) map[`${r.stationId}:${r.operation}`] = r;
  const polls = {};
  for (const p of s.readinessPolls) polls[`${p.stationId}:${p.operation}`] = p;
  const stations = new Set([s.fromStation, s.toStation, ...extraStations]);
  if (s.currentOrder) { if (s.currentOrder.fromStationId) stations.add(s.currentOrder.fromStationId);
                        if (s.currentOrder.toStationId) stations.add(s.currentOrder.toStationId); }
  for (const r of s.readiness) stations.add(r.stationId);
  for (const p of s.readinessPolls) stations.add(p.stationId);
  const now = Date.now();
  const key = JSON.stringify([...stations, s.readiness, s.readinessPolls.map(p => [p.stationId, p.operation, p.count, p.lastStatus])]);
  if (key === readyKey) return;
  readyKey = key;
  let h = `<div class="rd"><span class="dim">station</span><span class="dim">LOAD (파지 전)</span><span class="dim">UNLOAD (배치 전)</span>`;
  for (const st of stations) {
    h += `<span class="st">${st}</span>`;
    for (const op of ["LOAD", "UNLOAD"]) {
      const r = map[`${st}:${op}`]; const status = r ? r.status : "READY";
      const p = polls[`${st}:${op}`];
      const polling = p && (now - new Date(p.lastAt).getTime()) < 6000 && p.lastStatus !== "READY";
      h += `<span class="op">
        <button class="${status==='READY'?'okbtn':'off'}" onclick="setReadiness('${st}','${op}','READY')">READY</button>
        <button class="${status==='NOT_READY'?'danger':'off'}" onclick="setReadiness('${st}','${op}','NOT_READY')">NOT_READY</button>
        ${r && r.reasonCode ? `<span class="dim">${r.reasonCode}</span>` : ""}
        ${polling ? `<span class="poll">⏳ 로봇 대기 중 · ${p.count}회 · ${ts(p.lastAt)}</span>` :
          p ? `<span class="dim" style="font-size:11px">마지막 조회 ${ts(p.lastAt)} → ${p.lastStatus}</span>` : ""}
      </span>`;
    }
  }
  $("readiness").innerHTML = h + `</div>`;
}

function syncForm(s) {
  if (!formInit) {
    $("f_fromStationId").value = s.fromStation;
    $("f_toStationId").value = s.toStation;
    formInit = true;
  }
  if (document.activeElement !== $("auto")) $("auto").checked = !!s.autoDispatch;
  const st = s.serverState;
  $("btnSend").disabled = !(st === "READY" || st === "COOLDOWN");
  $("btnCancel").disabled = st !== "RUNNING";
  $("btnResume").disabled = st !== "HALTED";
  $("f_wcsOrderId").placeholder = `비우면 WCS-${new Date().toISOString().slice(0,10).replace(/-/g,"")}-${String(s.nextSeq).padStart(6,"0")}`;
  $("f_carrierId").placeholder = `비우면 ${s.carrierPrefix}-${String(s.nextSeq).padStart(6,"0")}`;
}

async function tick() {
  let s;
  try {
    const r = await fetch("/api/test/state", {cache:"no-store"});
    if (!r.ok) throw new Error("HTTP " + r.status);
    s = await r.json();
  } catch (e) {
    document.getElementById("dot").style.background = "var(--bad)";
    document.getElementById("meta").textContent = "서버 조회 실패: " + e.message;
    return;
  }
  const serials = Object.keys(s.latest);
  const d = serials.length ? s.latest[serials[0]] : null;
  const p = d && d.payloadJson ? JSON.parse(d.payloadJson) : {};
  const age = d ? (Date.now() - new Date(d.receivedAt).getTime()) / 1000 : null;
  const live = age !== null && age < 3;

  document.getElementById("dot").style.background = live ? "var(--ok)" : "var(--warn)";
  document.getElementById("meta").innerHTML = d
    ? `${d.robotSerial} · ${d.robotType} · 최종 수신 ${ts(d.receivedAt)} (<b class="${live?'ok':'warn'}">${age.toFixed(1)}초 전</b>) · 누적 ${s.received}건`
    : `아직 status 수신 없음 · 서버 ${s.serverState}`;

  let b = "";
  if (s.serverState === "HALTED") {
    const e = s.lastError || {};
    b += `<div class="banner err">오더 실패 — 발행 중단 (${e.wcsOrderId ?? "—"}, ${ts(e.receivedAt)})<br>
          <span style="font-weight:400">message: ${e.message ? e.message : "<i>(없음)</i>"}</span>
          <button onclick="resume()" style="margin-left:12px">재개</button></div>`;
  }
  if (s.robotReachable === false) b += `<div class="banner warn">로봇 오더 서버(${s.robotUrl})에 연결할 수 없습니다 — 데모가 DRY_RUN=0으로 떠 있는지, 주소/포트가 맞는지 확인</div>`;
  if (d && d.sourceIsStale) b += `<div class="banner warn">isStale=true — 로봇 측 업로더가 SDK 상태를 2초 이상 못 받고 있습니다</div>`;
  if (d && !live) b += `<div class="banner warn">${age.toFixed(0)}초간 새 status가 없습니다 — 데모/업로더 동작 확인</div>`;
  document.getElementById("banner").innerHTML = b;

  syncForm(s);
  renderReadiness(s);
  const waiting = s.readinessPolls.filter(p => p.lastStatus !== "READY" && (Date.now() - new Date(p.lastAt).getTime()) < 6000);
  if (waiting.length) {
    document.getElementById("banner").innerHTML += waiting.map(p =>
      `<div class="banner warn">로봇이 ${p.stationId} ${p.operation} readiness를 기다리는 중 (${p.count}회 조회, ${p.lastStatus}) — PIO 카드에서 READY로 바꾸면 진행</div>`).join("");
  }
  const o = s.currentOrder;
  document.getElementById("server").innerHTML =
    `<div class="big ${STATE_CLS[s.serverState] || ''}">${s.serverState}</div>
     <div class="row"><span class="k">발행 모드</span><span class="v ${s.autoDispatch?'ok':'info'}">${s.autoDispatch ? "자동" : "수동"}</span></div>
     <div class="row"><span class="k">로봇 오더 서버</span><span class="v ${s.robotReachable===false?'bad':s.robotReachable?'ok':'dim'}">${s.robotUrl}</span></div>
     <div class="row"><span class="k">현재 오더</span><span class="v">${o ? o.wcsOrderId : "—"}</span></div>
     <div class="row"><span class="k">from → to</span><span class="v">${o ? `${o.fromStationId} → ${o.toStationId}` : "—"}</span></div>
     <div class="row"><span class="k">오더 상태</span><span class="v">${o ? o.orderStatus : "—"}</span></div>
     <div class="row"><span class="k">발행 시각</span><span class="v">${o ? ts(o.acceptedAt) : "—"}</span></div>
     <div class="row"><span class="k">대기 잔여</span><span class="v ${s.cooldownRemaining!==null?'warn':''}">${
        s.cooldownRemaining !== null ? f(s.cooldownRemaining,1) + " / " + f(s.cooldownSec,0) + " s" : "—"}</span></div>`;

  document.getElementById("orderlog").innerHTML = s.orderLog.length
    ? `<table><tr><th>wcsOrderId</th><th>carrier</th><th>상태</th><th>발행</th></tr>` +
      s.orderLog.map(c => `<tr><td>${c.wcsOrderId}</td><td>${c.carrierId}</td><td>${c.orderStatus}</td><td>${ts(c.acceptedAt)}</td></tr>`).join("") + `</table>`
    : `<div class="dim">아직 발행한 오더 없음 ([오더 발행] 또는 자동 발행 ON)</div>`;

  document.getElementById("eventlog").innerHTML = s.eventLog.length
    ? `<table><tr><th>wcsOrderId</th><th>type</th><th>result</th><th>message</th><th>수신</th></tr>` +
      s.eventLog.map(e => `<tr><td>${e.wcsOrderId ?? ""}</td><td class="${e.eventType==='FAILED'?'bad':e.eventType==='COMPLETED'?'ok':/^ARRIVED/.test(e.eventType)?'info':''}">${e.eventType ?? ""}</td><td>${e.result ?? ""}</td><td style="text-align:left">${e.message ?? ""}</td><td>${ts(e.receivedAt)}</td></tr>`).join("") + `</table>`
    : `<div class="dim">아직 수신한 이벤트 없음</div>`;

  renderComm(s.commLog);

  if (!d) {
    for (const id of ["state","batt","pose","sys","enc","raw"]) document.getElementById(id).innerHTML = `<span class="dim">—</span>`;
    return;
  }

  document.getElementById("state").innerHTML =
    boolTile("비상정지", d.emergencyStop, true) +
    boolTile("전원", d.mainPowerOn) +
    boolTile("서보", d.servoOn) +
    boolTile("제어준비", d.controlReady) +
    `<div class="tile"><div class="k">작업</div><div class="v ${
       d.workCycle==='ERROR'?'bad':d.workCycle==='UNKNOWN'?'dim':d.workCycle==='WORKING'?'ok':'info'}">${d.workCycle}</div></div>` +
    (d.errorMessage ? `<div class="tile" style="flex-basis:100%;text-align:left"><div class="k">error_message</div><div class="v bad" style="font-size:13px">${d.errorMessage}</div></div>` : "");

  const pct = d.batteryPercent ?? 0;
  document.getElementById("batt").innerHTML =
    `<div class="big">${f(pct,1)}<span style="font-size:14px" class="dim"> %</span></div>
     <div class="bar"><i style="width:${Math.max(0,Math.min(100,pct))}%;background:${
       pct<20?'var(--bad)':pct<40?'var(--warn)':'var(--ok)'}"></i></div>
     <div class="row"><span class="k">전압</span><span class="v">${f(d.batteryVoltage)} V</span></div>
     <div class="row"><span class="k">전류</span><span class="v">${f(d.batteryCurrent)} A</span></div>`;

  document.getElementById("pose").innerHTML = (d.poseX === null && d.poseY === null)
    ? `<div class="dim">pose 없음 (null)</div>`
    : `<div class="row"><span class="k">x</span><span class="v">${f(d.poseX,3)} m</span></div>
       <div class="row"><span class="k">y</span><span class="v">${f(d.poseY,3)} m</span></div>
       <div class="row"><span class="k">rz</span><span class="v">${f(d.poseRz,3)} rad / ${f(deg(d.poseRz),1)}°</span></div>`;

  const up = d.uptimeSeconds ?? 0;
  document.getElementById("sys").innerHTML =
    `<div class="row"><span class="k">CPU</span><span class="v">${f(d.cpuUsage,1)} %</span></div>
     <div class="row"><span class="k">메모리</span><span class="v">${f(d.memoryUsage,1)} %</span></div>
     <div class="row"><span class="k">가동시간</span><span class="v">${
       Math.floor(up/3600)}h ${Math.floor(up%3600/60)}m ${Math.floor(up%60)}s</span></div>`;

  const enc = p.encoder || {};
  let rows = `<table><tr><th>관절</th><th>rad</th><th>deg</th></tr>`;
  for (const [key, [ko, names]] of Object.entries(PARTS)) {
    const vals = enc[key] || [];
    rows += `<tr><td class="grp" colspan="3">${ko} <span class="dim">(${key}[${vals.length}])</span></td></tr>`;
    names.forEach((n, i) => {
      const v = vals[i];
      rows += `<tr><td>${i}. ${n}</td><td>${f(v,5)}</td><td>${v===undefined?'—':f(deg(v),1)}°</td></tr>`;
    });
  }
  document.getElementById("enc").innerHTML = rows + `</table>`;
  document.getElementById("raw").textContent = JSON.stringify(p, null, 2);
}
tick(); setInterval(tick, 1000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        path = url.path

        if path in ("/", "/index.html"):
            return self._send(200, "text/html; charset=utf-8", PAGE.encode())
        if path == "/health":
            return self._send(200, "text/plain; charset=utf-8", b"healthy")
        if path == "/api/test/state":
            return self._json(200, SIM.dashboard_state())

        if path.startswith(STATIONS_PREFIX + "/"):
            parts = path[len(STATIONS_PREFIX) + 1:].strip("/").split("/")
            if len(parts) == 2 and parts[1] == "readiness":
                query = parse_qs(url.query)
                operation = (query.get("operation", [""])[0] or "").strip().upper()
                if operation not in READINESS_OPERATIONS:
                    return self._json(400, {"errorCode": "INVALID_REQUEST",
                                            "message": f"operation must be one of {', '.join(READINESS_OPERATIONS)}",
                                            "timestamp": _now_iso()})
                order_id = (query.get("wcsOrderId", [""])[0] or "").strip() or None
                return self._json(200, SIM.get_readiness(unquote(parts[0]), operation, order_id))

        if path.startswith(STATUS_PREFIX + "/"):
            parts = path[len(STATUS_PREFIX) + 1:].strip("/").split("/")
            if len(parts) == 2:
                serial, view = unquote(parts[0]), parts[1]
                if view == "latest":
                    record = SIM.latest(serial)
                    if record is None:
                        return self._json(404, {"error": "no status yet", "robotSerial": serial})
                    return self._json(200, record)
                if view == "history":
                    limit = int(parse_qs(url.query).get("limit", ["100"])[0])
                    return self._json(200, SIM.history(serial, limit))

        self._json(404, {"error": "not found", "path": path})

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path

        if path == "/api/test/resume":
            SIM.resume()
            return self._json(200, {"ok": True, "serverState": SIM.dashboard_state()["serverState"]})

        if path == "/api/test/cancel":
            ok, message = SIM.cancel_current()
            return self._json(200 if ok else 409,
                              {"ok": ok, "message": message,
                               "serverState": SIM.dashboard_state()["serverState"]})

        if path in ("/api/test/order", "/api/test/auto", "/api/test/readiness"):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else ""
            try:
                fields = json.loads(raw) if raw.strip() else {}
            except ValueError as error:
                return self._json(400, {"ok": False, "message": f"invalid JSON: {error}"})
            if not isinstance(fields, dict):
                return self._json(400, {"ok": False, "message": "body must be an object"})
            if path == "/api/test/auto":
                enabled = SIM.set_auto(bool(fields.get("enabled")))
                return self._json(200, {"ok": True, "autoDispatch": enabled})
            if path == "/api/test/readiness":
                station = str(fields.get("stationId") or "").strip()
                operation = str(fields.get("operation") or "").strip().upper()
                status = str(fields.get("status") or "").strip().upper()
                if not station or operation not in READINESS_OPERATIONS or status not in ("READY", "NOT_READY"):
                    return self._json(400, {"ok": False, "message": "stationId, operation(LOAD|UNLOAD), "
                                                                    "status(READY|NOT_READY) 필요"})
                entry = SIM.set_readiness(station, operation, status, fields.get("reasonCode"))
                return self._json(200, {"ok": True, "readiness": entry})
            ok, message, code = SIM.dispatch_manual(fields)
            state = SIM.dashboard_state()
            return self._json(code, {"ok": ok, "message": message,
                                     "serverState": state["serverState"],
                                     "currentOrder": state["currentOrder"]})

        if path == EVENT_PATH or path == STATUS_PREFIX or path.startswith(STATUS_PREFIX + "/"):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as error:
                return self._json(400, {"accepted": False, "error": f"invalid JSON: {error}"})
            if not isinstance(payload, dict):
                return self._json(400, {"accepted": False, "error": "body must be an object"})

            if path == EVENT_PATH:
                if not payload.get("wcsOrderId") or not payload.get("eventType"):
                    return self._json(400, {"accepted": False, "error": "wcsOrderId and eventType required"})
                return self._json(200, SIM.on_transport_event(payload))

            if not payload.get("robotSerial"):
                return self._json(400, {"accepted": False, "error": "robotSerial required"})
            record = SIM.on_status(payload)
            return self._json(201, {"accepted": True, "recordId": record["recordId"],
                                    "receivedAt": record["receivedAt"]})

        self._json(404, {"error": "not found", "path": path})

    def _json(self, code: int, body: Any) -> None:
        self._send(code, "application/json; charset=utf-8",
                   json.dumps(body, ensure_ascii=False).encode())

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):  # 1Hz status/대시보드 폴링이라 접근 로그는 끈다 (전이·오더·이벤트는 SIM이 남김)
        pass


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    log.info("가짜 WCS 서버: http://%s:%d  (대시보드 http://localhost:%d, 완료 후 대기 %.0f초)",
             BIND, PORT, PORT, COOLDOWN_SEC)
    log.info("오더 발행 대상(로봇): POST %s%s  (%s -> %s)", ROBOT_URL, ROBOT_ORDER_PATH, FROM_STATION, TO_STATION)
    log.info("발행 모드: %s", "자동 (READY마다 발행)" if AUTO_DISPATCH else "수동 (대시보드 [오더 발행] 또는 POST /api/test/order)")
    log.info("로봇 측: WCS_BASE_URL=http://<이 PC IP>:%d DRY_RUN=0 python demo_full_sequence_loop.py", PORT)

    stop = threading.Event()
    threading.Thread(target=SIM.run_dispatcher, args=(stop,), name="wcs-dispatcher", daemon=True).start()
    try:
        ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        log.info("종료")
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
