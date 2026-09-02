"""내부 테스트용 가짜 SFA WCS 서버 (표준 라이브러리만 사용).

    python tools/fake_wcs/server.py                 # http://0.0.0.0:5224
    FAKE_WCS_PORT=5224 FAKE_WCS_COOLDOWN_SEC=15 python tools/fake_wcs/server.py

데모(demo_full_sequence_loop.py)와 호환되는 엔드포인트를 제공하고, 아래 시퀀스로 작업 명령을 낸다.

    READY ──(로봇 폴링)──> START 발행 ──> RUNNING
    RUNNING ──(WORKING 확인 후 DONE 수신)──> COOLDOWN(15초) ──(만료 + 폴링)──> START 발행 ──> RUNNING
    RUNNING/COOLDOWN ──(ERROR 수신)──> HALTED ──(대시보드 [재개] / POST /api/test/resume)──> READY

엔드포인트 (실 WCS 호환):
    GET  /health                                    -> "healthy"
    POST /api/v1/rb/rby1/status                     -> 201 {"accepted": true, "recordId": ...}
    GET  /api/v1/rb/rby1/status/{serial}/latest
    GET  /api/v1/rb/rby1/status/{serial}/history?limit=N
테스트 전용:
    GET  /api/v1/rb/rby1/command/{serial}           -> {"commandId": n, "command": "START"|"WAIT", "issuedAt": ...}
    POST /api/test/resume                           -> HALTED 해제
    GET  /api/test/state                            -> 대시보드용 상태 JSON
    GET  /                                          -> 대시보드
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

PORT = int(os.getenv("FAKE_WCS_PORT", "5224"))
BIND = os.getenv("FAKE_WCS_BIND", "0.0.0.0")
COOLDOWN_SEC = float(os.getenv("FAKE_WCS_COOLDOWN_SEC", "15"))
HISTORY_MAX = 1000
COMMAND_LOG_MAX = 100

STATUS_PREFIX = "/api/v1/rb/rby1/status"
COMMAND_PREFIX = "/api/v1/rb/rby1/command"

log = logging.getLogger("fake-wcs")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _str_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


class WcsSimulator:
    """수신 status 저장 + 작업 명령 상태 머신. 모든 상태는 lock 하나로 보호한다."""

    def __init__(self, cooldown_sec: float) -> None:
        self._lock = threading.Lock()
        self._cooldown_sec = cooldown_sec
        self._state = "READY"
        self._command_id = 0
        self._current_command: dict[str, Any] | None = None
        self._seen_working = False
        self._cooldown_until: float | None = None
        self._last_error: dict[str, Any] | None = None
        self._history: dict[str, deque[dict[str, Any]]] = {}
        self._command_log: deque[dict[str, Any]] = deque(maxlen=COMMAND_LOG_MAX)
        self._received = 0

    # ── status 수신 ─────────────────────────────────────────────
    def on_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = _flatten(payload)
        serial = record["robotSerial"]
        cycle = record["workCycle"]

        with self._lock:
            self._received += 1
            self._history.setdefault(serial, deque(maxlen=HISTORY_MAX)).appendleft(record)

            if cycle == "ERROR" and self._state != "HALTED":
                self._last_error = {
                    "robotSerial": serial,
                    "message": record["errorMessage"],
                    "receivedAt": record["receivedAt"],
                    "commandId": self._command_id,
                }
                self._transition("HALTED", f"ERROR 수신: {record['errorMessage']!r}")
            elif self._state == "RUNNING":
                if cycle == "WORKING":
                    self._seen_working = True
                elif cycle == "DONE" and self._seen_working:
                    self._cooldown_until = time.monotonic() + self._cooldown_sec
                    self._transition("COOLDOWN", f"DONE 수신 (commandId={self._command_id}), {self._cooldown_sec:.0f}초 대기")

        if self._received % 10 == 0:
            log.info("status 수신 %d건 (최근: serial=%s cycle=%s)", self._received, serial, cycle)
        return record

    # ── 명령 조회 ───────────────────────────────────────────────
    def get_command(self, serial: str) -> dict[str, Any]:
        with self._lock:
            if self._state == "COOLDOWN" and self._cooldown_until is not None \
                    and time.monotonic() >= self._cooldown_until:
                self._cooldown_until = None
                self._transition("READY", "대기 시간 만료")

            if self._state == "READY":
                return self._issue_start(serial)
            if self._state == "RUNNING" and self._current_command is not None:
                return dict(self._current_command)

            return {"commandId": self._command_id, "command": "WAIT", "issuedAt": _now_iso()}

    def _issue_start(self, serial: str) -> dict[str, Any]:
        self._command_id += 1
        self._seen_working = False
        self._current_command = {"commandId": self._command_id, "command": "START", "issuedAt": _now_iso()}
        self._command_log.appendleft({**self._current_command, "robotSerial": serial})
        self._transition("RUNNING", f"START 발행 (commandId={self._command_id}, serial={serial})")
        return dict(self._current_command)

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
            records = self._history.get(serial, ())
            return [dict(r) for r in list(records)[:limit]]

    def dashboard_state(self) -> dict[str, Any]:
        with self._lock:
            remaining = None
            if self._state == "COOLDOWN" and self._cooldown_until is not None:
                remaining = max(0.0, self._cooldown_until - time.monotonic())
            latest = {serial: dict(records[0]) for serial, records in self._history.items() if records}
            return {
                "serverState": self._state,
                "cooldownSec": self._cooldown_sec,
                "cooldownRemaining": remaining,
                "currentCommand": dict(self._current_command) if self._current_command else None,
                "seenWorking": self._seen_working,
                "lastError": dict(self._last_error) if self._last_error else None,
                "commandLog": list(self._command_log),
                "received": self._received,
                "latest": latest,
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


SIM = WcsSimulator(COOLDOWN_SEC)


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
  .row { display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid var(--line); }
  .row:last-child { border-bottom:0; }
  .row .k { color:var(--dim); }
  .row .v { font-variant-numeric:tabular-nums; font-weight:600; }
  .bar { height:9px; background:#000; border-radius:5px; overflow:hidden; margin:8px 0 4px; }
  .bar > i { display:block; height:100%; background:var(--ok); transition:width .4s; }
  .banner { padding:10px 12px; border-radius:8px; margin-bottom:12px; font-weight:600; }
  .banner.err { background:#3d1418; border:1px solid var(--bad); color:#ffb4ae; }
  .banner.warn{ background:#3a2d10; border:1px solid var(--warn); color:#f0d48a; }
  .banner.info{ background:#10233a; border:1px solid var(--info); color:#9ccbff; }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  th,td { text-align:right; padding:3px 6px; border-bottom:1px solid var(--line); font-size:12px; }
  th:first-child, td:first-child { text-align:left; }
  th { color:var(--dim); font-weight:600; }
  .grp { color:var(--info); font-weight:700; padding-top:8px; }
  .full { grid-column:1/-1; }
  button { background:var(--info); color:#000; border:0; border-radius:6px; padding:6px 14px;
           font-weight:700; cursor:pointer; margin-left:12px; }
  button:disabled { background:var(--line); color:var(--dim); cursor:default; }
  .big { font-size:26px; font-weight:800; }
  #dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--dim);
         margin-right:5px; vertical-align:middle; }
  pre { background:#000; padding:10px; border-radius:8px; font-size:11px; overflow:auto; max-height:260px; margin:0; }
</style></head><body>
<h1><span id="dot"></span>가짜 WCS — RBY1 테스트 서버</h1>
<div class="sub" id="meta">연결 중…</div>
<div id="banner"></div>
<div class="grid">
  <div class="card"><h2>서버 상태 / 명령</h2><div id="server"></div></div>
  <div class="card"><h2>로봇 상태</h2><div class="tiles" id="state"></div></div>
  <div class="card"><h2>배터리</h2><div id="batt"></div></div>
  <div class="card"><h2>위치 (오도메트리)</h2><div id="pose"></div></div>
  <div class="card"><h2>제어 PC</h2><div id="sys"></div></div>
  <div class="card"><h2>명령 이력</h2><div id="cmdlog"></div></div>
  <div class="card full"><h2>엔코더 — 관절 (rad / deg)</h2><div id="enc"></div></div>
  <div class="card full"><h2>마지막 수신 payload (원본)</h2><pre id="raw"></pre></div>
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
const STATE_CLS = {READY:"info", RUNNING:"ok", COOLDOWN:"warn", HALTED:"bad"};
const boolTile = (k, on, invert=false) => {
  const good = invert ? !on : on;
  return `<div class="tile"><div class="k">${k}</div>
          <div class="v ${good?'ok':'bad'}">${on?'ON':'OFF'}</div></div>`;
};

async function resume() {
  await fetch("/api/test/resume", {method:"POST"});
  tick();
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
    b += `<div class="banner err">로봇 ERROR 수신 — 명령 발행 중단 (commandId=${e.commandId ?? "—"}, ${ts(e.receivedAt)})<br>
          <span style="font-weight:400">error_message: ${e.message ? e.message : "<i>(없음)</i>"}</span>
          <button onclick="resume()">재개</button></div>`;
  }
  if (d && d.sourceIsStale) b += `<div class="banner warn">isStale=true — 로봇 측 업로더가 SDK 상태를 2초 이상 못 받고 있습니다</div>`;
  if (d && !live) b += `<div class="banner warn">${age.toFixed(0)}초간 새 status가 없습니다 — 데모/업로더 동작 확인</div>`;
  document.getElementById("banner").innerHTML = b;

  const cmd = s.currentCommand;
  document.getElementById("server").innerHTML =
    `<div class="big ${STATE_CLS[s.serverState] || ''}">${s.serverState}</div>
     <div class="row"><span class="k">현재 명령</span><span class="v">${cmd ? `${cmd.command} #${cmd.commandId}` : "—"}</span></div>
     <div class="row"><span class="k">발행 시각</span><span class="v">${cmd ? ts(cmd.issuedAt) : "—"}</span></div>
     <div class="row"><span class="k">WORKING 확인</span><span class="v ${s.seenWorking?'ok':'dim'}">${s.seenWorking ? "예" : "아니오"}</span></div>
     <div class="row"><span class="k">대기 잔여</span><span class="v ${s.cooldownRemaining!==null?'warn':''}">${
        s.cooldownRemaining !== null ? f(s.cooldownRemaining,1) + " / " + f(s.cooldownSec,0) + " s" : "—"}</span></div>`;

  document.getElementById("cmdlog").innerHTML = s.commandLog.length
    ? `<table><tr><th>#</th><th>명령</th><th>serial</th><th>발행</th></tr>` +
      s.commandLog.map(c => `<tr><td>${c.commandId}</td><td>${c.command}</td><td>${c.robotSerial}</td><td>${ts(c.issuedAt)}</td></tr>`).join("") + `</table>`
    : `<div class="dim">아직 발행한 명령 없음 (로봇이 폴링하면 START 발행)</div>`;

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

        if path.startswith(COMMAND_PREFIX + "/"):
            serial = unquote(path[len(COMMAND_PREFIX) + 1:].strip("/"))
            return self._json(200, SIM.get_command(serial))

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

        if path == STATUS_PREFIX or path.startswith(STATUS_PREFIX + "/"):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as error:
                return self._json(400, {"accepted": False, "error": f"invalid JSON: {error}"})
            if not isinstance(payload, dict) or not payload.get("robotSerial"):
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

    def log_message(self, *_):  # 1Hz status/명령 폴링이라 접근 로그는 끈다 (전이/발행은 SIM이 남김)
        pass


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    log.info("가짜 WCS 서버: http://%s:%d  (대시보드 http://localhost:%d, 완료 후 대기 %.0f초)",
             BIND, PORT, PORT, COOLDOWN_SEC)
    log.info("로봇 측: WCS_BASE_URL=http://<이 PC IP>:%d DRY_RUN=0 python demo_full_sequence_loop.py", PORT)
    try:
        ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        log.info("종료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
