"""최신 RBY1 상태와 작업 상태를 백그라운드에서 SFA WCS로 전송한다."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

from . import config
from .client import WcsClient
from .order_server import OrderReceiver
from .payload import build_status_payload
from .state import WcsStateStore

log = logging.getLogger(__name__)


class WcsPublisher:
    """RobotState는 빠르게 캐시하고, 최신 상태를 지정된 주기로 WCS에 전송한다."""

    def __init__(
        self,
        robot_model: Any,
        *,
        serial: str = config.ROBOT_SERIAL,
        upload_rate_hz: float = config.UPLOAD_RATE_HZ,
        dry_run: bool = config.DRY_RUN,
        client: WcsClient | None = None,
    ) -> None:
        if upload_rate_hz <= 0:
            raise ValueError("upload_rate_hz는 0보다 커야 합니다.")

        self._serial = serial
        self._period = 1.0 / upload_rate_hz
        self._dry_run = dry_run
        self._state = WcsStateStore(robot_model)
        self._client = client or WcsClient(
            config.WCS_BASE_URL,
            config.WCS_HEALTH_PATH,
            config.WCS_STATUS_PATH,
            config.HTTP_TIMEOUT_SEC,
            dry_run=dry_run,
            transport_event_path=config.WCS_TRANSPORT_EVENT_PATH,
        )
        # WCS → 로봇 반송 오더 수신 (AMR Transport Order 규격 준용). DRY_RUN이면 띄우지 않는다.
        self._order_receiver = OrderReceiver(config.ROBOT_ORDER_BIND, config.ROBOT_ORDER_PORT,
                                             config.ROBOT_ORDER_PATH)
        self._current_order: dict[str, Any] | None = None
        self._dry_run_order_seq = 0

        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._work_event_lock = threading.Lock()
        self._work_events: deque[tuple[str, str | None]] = deque()
        self._thread: threading.Thread | None = None

        self._sent = 0
        self._failed = 0
        self._waiting_logged = False
        self._state_error_logged = False

    def start(self) -> None:
        """WCS 연결을 확인하고 백그라운드 전송 스레드를 시작한다."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return

            if not self._dry_run and not self._client.check_health():
                raise ConnectionError("WCS 개발서버에 연결할 수 없습니다.")

            self._stop_event.clear()
            self._wake_event.clear()
            self._sent = 0
            self._failed = 0
            self._thread = threading.Thread(target=self._run, name="wcs-publisher", daemon=True)
            self._thread.start()
            self._wake_event.set()

            if not self._dry_run:
                self._order_receiver.start()

        log.info("WCS publisher 시작: %.1f Hz, serial=%s", 1.0 / self._period, self._serial)

    def stop(self, *, flush: bool = True) -> None:
        """전송 스레드를 종료하고, flush=True이면 마지막 상태를 한 번 더 전송한다."""
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return
            self._stop_event.set()
            self._wake_event.set()

        thread.join(timeout=config.HTTP_TIMEOUT_SEC + 1.0)
        if thread.is_alive():
            log.warning("WCS publisher 스레드가 제한시간 안에 종료되지 않았습니다.")
            return

        if flush:
            self._publish_due(force_current=True)

        with self._lifecycle_lock:
            if self._thread is thread:
                self._thread = None

        self._order_receiver.stop()
        log.info("WCS publisher 종료: sent=%d failed=%d", self._sent, self._failed)

    def on_state(self, state: Any, *_: Any) -> None:
        """SDK callback에서 호출한다. HTTP 통신 없이 최신 RobotState만 캐시한다."""
        first_state = self._state.latest_robot() is None

        try:
            self._state.update_robot_state(state)
            self._state_error_logged = False
        except Exception:  # noqa: BLE001 - SDK callback을 WCS 변환 오류로 중단하지 않는다.
            if not self._state_error_logged:
                log.exception("WCS용 RobotState 변환 실패")
                self._state_error_logged = True
            return

        if first_state:
            self._wake_event.set()

    def set_work_state(self, cycle: str, error_message: str | None = None) -> None:
        """작업 상태를 갱신하고 순서 보존 큐에 넣어 가능한 즉시 전송한다."""
        self._state.set_work_state(cycle, error_message)
        work_event = self._state.get_work_state()

        with self._work_event_lock:
            if not self._work_events or self._work_events[-1] != work_event:
                self._work_events.append(work_event)

        self._wake_event.set()

    # ── 반송 오더 (WCS → 로봇 push, 로봇 → WCS 이벤트 콜백) ──────────────────
    def wait_for_order(
        self,
        *,
        timeout: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> dict[str, Any] | None:
        """WCS가 POST한 반송 오더를 하나 꺼내 현재 오더로 잡는다.

        호출 스레드에서 블록한다. timeout 초과 또는 stop_event가 set되면 None.
        DRY_RUN이면 서버 없이도 데모가 돌도록 가짜 오더를 즉시 돌려준다.
        """
        if self._dry_run:
            self._dry_run_order_seq += 1
            order = {"wcsOrderId": f"DRYRUN-{self._dry_run_order_seq:06d}", "carrierId": None,
                     "fromStationId": "DRYRUN_FROM", "toStationId": "DRYRUN_TO", "priority": 0,
                     "timestamp": _now_iso()}
            log.info("[DRY_RUN] 반송 오더 대기 생략 -> %s", order["wcsOrderId"])
        else:
            order = self._order_receiver.wait_for_order(timeout=timeout, stop_event=stop_event)
            if order is None:
                return None
            log.info("반송 오더 시작: %s (%s -> %s)", order["wcsOrderId"],
                     order.get("fromStationId"), order.get("toStationId"))

        self._current_order = order
        return dict(order)

    def current_order(self) -> dict[str, Any] | None:
        return dict(self._current_order) if self._current_order else None

    def complete_order(self) -> bool:
        """현재 오더를 COMPLETED로 WCS에 보고한다."""
        return self._report_order_event("COMPLETED", "SUCCESS", None)

    def fail_order(self, message: str | None) -> bool:
        """현재 오더를 FAILED로 WCS에 보고한다. 진행 중인 오더가 없으면 아무것도 하지 않는다."""
        return self._report_order_event("FAILED", "FAILED", message)

    def _report_order_event(self, event_type: str, result: str, message: str | None,
                            attempts: int = 3) -> bool:
        order = self._current_order
        if order is None:
            log.debug("보고할 진행 중 오더가 없습니다 (%s)", event_type)
            return False

        order_id = str(order["wcsOrderId"])
        body = {
            "eventId": uuid.uuid4().hex,
            "wcsOrderId": order_id,
            "eventType": event_type,
            "robotSerial": self._serial,
            "result": result,
            "message": (message or "").strip()[:200] or None,
            "occurredAt": _now_iso(),
        }

        # WCS가 eventId로 멱등 처리하므로 같은 eventId로 몇 번 재시도해도 안전하다.
        succeeded = False
        for attempt in range(1, attempts + 1):
            if self._client.post_transport_event(body):
                succeeded = True
                break
            if attempt < attempts:
                time.sleep(1.0)

        if not succeeded:
            log.error("transport-event 보고 실패: %s %s (%d회 시도)", event_type, order_id, attempts)

        self._order_receiver.set_status(order_id, event_type)
        self._current_order = None
        return succeeded

    def _run(self) -> None:
        next_upload = time.monotonic()

        while not self._stop_event.is_set():
            timeout = max(0.0, next_upload - time.monotonic())
            self._wake_event.wait(timeout)
            self._wake_event.clear()

            if self._stop_event.is_set():
                break

            try:
                self._publish_due(force_current=False)
            except Exception:  # noqa: BLE001 - 전송 스레드는 다음 heartbeat에서 계속 재시도한다.
                log.exception("WCS publisher 반복 처리 실패")

            next_upload = time.monotonic() + self._period

    def _publish_due(self, *, force_current: bool) -> None:
        with self._publish_lock:
            event_attempted = self._drain_work_events()
            if force_current and not self._has_pending_work_event():
                self._publish_current()
            elif not event_attempted:
                self._publish_current()

    def _drain_work_events(self) -> bool:
        """상태 전환을 순서대로 보내고 실패한 항목은 다음 주기에 재시도한다."""
        attempted = False

        while True:
            with self._work_event_lock:
                if not self._work_events:
                    return attempted
                work_cycle, error_message = self._work_events[0]

            snapshot = self._state.latest_robot()
            if snapshot is None:
                self._log_waiting_for_state()
                return attempted

            attempted = True
            if not self._post(snapshot, work_cycle, error_message):
                return attempted

            with self._work_event_lock:
                self._work_events.popleft()

    def _publish_current(self) -> bool:
        snapshot = self._state.latest_robot()
        if snapshot is None:
            self._log_waiting_for_state()
            return False

        work_cycle, error_message = self._state.get_work_state()
        return self._post(snapshot, work_cycle, error_message)

    def _post(self, snapshot: Any, work_cycle: str, error_message: str | None) -> bool:
        self._waiting_logged = False

        try:
            body = build_status_payload(
                self._serial,
                snapshot,
                work_cycle=work_cycle,
                error_message=error_message,
            )
            succeeded = self._client.post_status(self._serial, body)
        except Exception:  # noqa: BLE001 - payload 오류도 데모 실행을 중단하지 않는다.
            log.exception("WCS status 생성 또는 전송 실패")
            succeeded = False

        if succeeded:
            self._sent += 1
        else:
            self._failed += 1

        if (self._sent + self._failed) % 10 == 0:
            log.info("WCS 전송 통계: sent=%d failed=%d", self._sent, self._failed)

        return succeeded

    def _has_pending_work_event(self) -> bool:
        with self._work_event_lock:
            return bool(self._work_events)

    def _log_waiting_for_state(self) -> None:
        if not self._waiting_logged:
            log.info("첫 RobotState를 기다리는 중입니다.")
            self._waiting_logged = True


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
