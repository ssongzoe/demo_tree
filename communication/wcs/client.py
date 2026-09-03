"""SFA WCS health 확인과 RBY1 status POST를 담당하는 HTTP 클라이언트."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote

import requests

log = logging.getLogger(__name__)


class WcsClient:
    def __init__(
        self,
        base_url: str,
        health_path: str,
        status_path: str,
        timeout: float,
        *,
        dry_run: bool = False,
        session: requests.Session | None = None,
        transport_event_path: str | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("WCS base_url이 비어 있습니다.")
        if timeout <= 0:
            raise ValueError("HTTP timeout은 0보다 커야 합니다.")

        self._base_url = base_url.rstrip("/")
        self._health_url = self._join_url(health_path)
        self._status_path = status_path
        self._timeout = timeout
        self._dry_run = dry_run
        self._logged_first_ok = False
        self._transport_event_path = transport_event_path
        self._event_error_logged = False
        self._owns_session = session is None
        self._session = session or requests.Session()
        self._session.headers.update({"Content-Type": "application/json; charset=utf-8"})

    def check_health(self) -> bool:
        """WCS /health가 명확한 healthy 응답을 반환할 때만 True다."""
        if self._dry_run:
            log.info("[DRY_RUN] WCS health check 생략: %s", self._health_url)
            return True

        try:
            response = self._session.get(self._health_url, timeout=self._timeout)
        except requests.RequestException as error:
            log.error("WCS health check 실패 (%s): %s", self._health_url, error)
            return False

        healthy = response.status_code == 200 and _is_healthy_body(response)
        log.info(
            "WCS health check: HTTP %s, body=%r -> %s",
            response.status_code,
            response.text.strip()[:80],
            "OK" if healthy else "NOT OK",
        )
        return healthy

    def post_status(self, serial: str, payload: dict[str, Any]) -> bool:
        """RBY1 status 한 건을 POST하고 2xx 응답일 때 True를 반환한다."""
        url = self._status_url(serial)

        if self._dry_run:
            log.info("[DRY_RUN] POST %s\n%s", url, json.dumps(payload, ensure_ascii=False, indent=2))
            return True

        try:
            response = self._session.post(url, json=payload, timeout=self._timeout)
        except requests.RequestException as error:
            log.warning("WCS status POST 실패: %s", error)
            return False

        if not 200 <= response.status_code < 300:
            log.warning(
                "WCS status POST -> HTTP %s: %s",
                response.status_code,
                response.text.strip()[:160],
            )
            return False

        if not self._logged_first_ok:
            self._logged_first_ok = True
            log.info(
                "WCS status POST OK -> HTTP %s %s",
                response.status_code,
                response.text.strip()[:200],
            )
        else:
            log.debug("WCS status POST -> HTTP %s", response.status_code)

        return True

    def post_transport_event(self, payload: dict[str, Any]) -> bool:
        """반송 완료/실패 이벤트 한 건을 WCS에 콜백한다 (AMR transport-events 규격 준용).

        payload: {eventId, wcsOrderId, eventType, robotSerial, result, message, occurredAt}
        2xx면 True. WCS는 eventId 기준으로 중복을 멱등 처리하므로 재시도해도 안전하다.
        """
        if self._transport_event_path is None:
            log.warning("transport_event_path가 설정되지 않아 이벤트를 보내지 않습니다.")
            return False

        url = self._join_url(self._transport_event_path)
        if self._dry_run:
            log.info("[DRY_RUN] POST %s\n%s", url, json.dumps(payload, ensure_ascii=False, indent=2))
            return True

        try:
            response = self._session.post(url, json=payload, timeout=self._timeout)
        except requests.RequestException as error:
            self._log_event_error("WCS transport-event POST 실패: %s", error)
            return False

        if not 200 <= response.status_code < 300:
            self._log_event_error("WCS transport-event POST -> HTTP %s: %s",
                                  response.status_code, response.text.strip()[:160])
            return False

        self._event_error_logged = False
        log.info("WCS transport-event OK: %s %s -> HTTP %s %s",
                 payload.get("eventType"), payload.get("wcsOrderId"),
                 response.status_code, response.text.strip()[:120])
        return True

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def _log_event_error(self, message: str, *args: Any) -> None:
        # 재시도 루프에서 같은 오류가 반복되면 첫 1회만 남긴다.
        if not self._event_error_logged:
            log.warning(message, *args)
            self._event_error_logged = True

    def _status_url(self, serial: str) -> str:
        encoded_serial = quote(serial, safe="")
        return self._join_url(self._status_path.format(serial=encoded_serial))

    def _join_url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"


def _is_healthy_body(response: requests.Response) -> bool:
    text = response.text.strip().lower()
    if text == "healthy":
        return True

    try:
        body = response.json()
    except ValueError:
        return False

    if not isinstance(body, dict):
        return False
    if body.get("healthy") is True:
        return True

    status = body.get("status", body.get("health", ""))
    return isinstance(status, str) and status.strip().lower() == "healthy"
