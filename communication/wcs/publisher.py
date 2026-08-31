"""최신 RBY1 상태와 작업 상태를 백그라운드에서 SFA WCS로 전송한다."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from . import config
from .client import WcsClient
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
        )

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
