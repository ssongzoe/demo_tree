"""WCS → RBY1 반송 오더 수신 서버.

AMR 쪽 SFA-WCS 규격(v07.2, `4. Transport Order 생성 API`)을 RBY1에 맞게 축소한 것이다.
WCS가 로봇 측으로 오더를 POST하고, 로봇은 즉시 ACCEPTED로 응답한 뒤 작업을 시작한다.
완료/실패는 WcsClient.post_transport_event()로 WCS에 콜백한다.

    POST {ROBOT_ORDER_PATH}
    { "wcsOrderId": "WCS-20260902-000001", "carrierId": "TOTE-000001",
      "fromStationId": "RACK01_PORT01", "toStationId": "CV02_IN",
      "priority": 5, "timestamp": "2026-09-02T09:00:00.000Z" }
    -> 201 { "wcsOrderId": ..., "orderStatus": "ACCEPTED", "timestamp": ... }
       200 동일 wcsOrderId·동일 내용 재전송 (현재 상태 반환, 멱등)
       409 DUPLICATE_ORDER_CONFLICT 동일 wcsOrderId·다른 내용
       400 INVALID_REQUEST wcsOrderId 누락 또는 JSON 형식 오류
             (carrierId/fromStationId/toStationId/priority/timestamp는 참조용이라 없어도 접수한다)
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# 오더에서 사용하는 필드. 이 외의 필드가 들어와도 접수는 하되 저장·사용하지 않는다.
ORDER_FIELDS = ("wcsOrderId", "carrierId", "fromStationId", "toStationId", "priority", "timestamp")
# 필수는 오더 식별자뿐이다. Station ID 등 나머지는 참조용이라 없어도 접수한다.
REQUIRED_FIELDS = ("wcsOrderId",)
# 멱등 판정에 쓰는 필드. timestamp는 재전송마다 달라질 수 있으므로 제외한다.
COMPARE_FIELDS = ("carrierId", "fromStationId", "toStationId", "priority")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class OrderReceiver:
    """오더를 받아 큐에 쌓고, 데모 루프가 wait_for_order()로 하나씩 꺼내 간다."""

    def __init__(self, bind: str, port: int, order_path: str) -> None:
        self._bind = bind
        self._port = port
        self._order_path = "/" + order_path.strip("/")
        self._lock = threading.Lock()
        self._arrived = threading.Condition(self._lock)
        self._orders: dict[str, dict[str, Any]] = {}
        self._queue: deque[dict[str, Any]] = deque()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ── 수명 ────────────────────────────────────────────────
    def start(self) -> None:
        if self._server is not None:
            return
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                if urlparse(self.path).path != receiver._order_path:
                    return self._json(404, {"errorCode": "NOT_FOUND", "path": self.path})
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as error:
                    return self._json(400, {"errorCode": "INVALID_REQUEST",
                                            "message": f"invalid JSON: {error}", "timestamp": _now_iso()})
                code, response = receiver.submit(body)
                self._json(code, response)

            def do_GET(self):  # noqa: N802
                if urlparse(self.path).path == "/health":
                    return self._send(200, "text/plain; charset=utf-8", b"healthy")
                self._json(404, {"errorCode": "NOT_FOUND", "path": self.path})

            def _json(self, code: int, body: Any) -> None:
                self._send(code, "application/json; charset=utf-8",
                           json.dumps(body, ensure_ascii=False).encode())

            def _send(self, code: int, ctype: str, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):  # 접근 로그는 submit()에서 의미 있는 것만 남긴다
                pass

        self._server = ThreadingHTTPServer((self._bind, self._port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="wcs-order-server", daemon=True)
        self._thread.start()
        log.info("반송 오더 수신 서버 시작: http://%s:%d%s", self._bind, self._port, self._order_path)

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info("반송 오더 수신 서버 종료")

    # ── 오더 ────────────────────────────────────────────────
    def submit(self, body: Any) -> tuple[int, dict[str, Any]]:
        """HTTP 핸들러에서 호출. (status code, response body)를 돌려준다."""
        if not isinstance(body, dict):
            return 400, {"errorCode": "INVALID_REQUEST", "message": "body must be an object",
                         "timestamp": _now_iso()}
        missing = [field for field in REQUIRED_FIELDS if not body.get(field)]
        if missing:
            return 400, {"errorCode": "INVALID_REQUEST", "message": f"missing fields: {', '.join(missing)}",
                         "timestamp": _now_iso()}

        order_id = str(body["wcsOrderId"])
        unknown = [key for key in body if key not in ORDER_FIELDS]
        if unknown:
            log.info("오더 %s: 사용하지 않는 필드는 무시합니다: %s", order_id, ", ".join(unknown))

        with self._lock:
            existing = self._orders.get(order_id)
            if existing is not None:
                same = all(existing["request"].get(f) == body.get(f) for f in COMPARE_FIELDS)
                if same:
                    log.info("동일 오더 재전송 (멱등 처리): %s -> %s", order_id, existing["status"])
                    return 200, self._accept_body(order_id, existing["status"])
                log.warning("동일 wcsOrderId에 다른 내용: %s", order_id)
                return 409, {"errorCode": "DUPLICATE_ORDER_CONFLICT",
                             "message": "The wcsOrderId already exists with different order data.",
                             "timestamp": _now_iso()}

            record = {"request": _use_fields(body), "status": "ACCEPTED", "acceptedAt": _now_iso()}
            self._orders[order_id] = record
            self._queue.append(record)
            self._arrived.notify_all()

        log.info("반송 오더 수신: %s (%s -> %s, carrier=%s)", order_id,
                 body.get("fromStationId"), body.get("toStationId"), body.get("carrierId"))
        return 201, self._accept_body(order_id, "ACCEPTED")

    def wait_for_order(self, timeout: float | None = None,
                       stop_event: threading.Event | None = None) -> dict[str, Any] | None:
        """큐에서 오더 하나를 꺼낸다. timeout 초과 또는 stop_event set이면 None."""
        deadline = None if timeout is None else datetime.now().timestamp() + timeout
        with self._lock:
            while not self._queue:
                if stop_event is not None and stop_event.is_set():
                    return None
                remaining = None if deadline is None else deadline - datetime.now().timestamp()
                if remaining is not None and remaining <= 0:
                    return None
                # stop_event도 봐야 하므로 최대 0.5초씩 끊어서 기다린다.
                self._arrived.wait(0.5 if remaining is None else min(0.5, remaining))
            record = self._queue.popleft()
            record["status"] = "IN_PROGRESS"
            return dict(record["request"])

    def set_status(self, order_id: str, status: str) -> None:
        with self._lock:
            record = self._orders.get(order_id)
            if record is not None:
                record["status"] = status

    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    @staticmethod
    def _accept_body(order_id: str, status: str) -> dict[str, Any]:
        return {"wcsOrderId": order_id, "orderStatus": status, "timestamp": _now_iso()}


def _use_fields(body: dict[str, Any]) -> dict[str, Any]:
    """오더에서 실제로 사용하는 필드만 추린다(정의 순서 유지)."""
    return {field: body[field] for field in ORDER_FIELDS if field in body}
