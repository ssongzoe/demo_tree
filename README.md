# demo_tree — RB-Y1 tote 이송 데모 + SFA WCS 연동

## 구성

| 경로 | 역할 |
|---|---|
| `demo_full_sequence_loop.py` | tote 인식/파지 → 이송 → AR 정렬 → 배치 → 복귀 반복 데모. WCS 상태 업로드 + 반송 오더 수신/완료 보고 |
| `communication/wcs/` | SFA WCS 통신 모듈 (status POST 1Hz, 오더 수신 서버, transport-events 콜백) |
| `tools/fake_wcs/server.py` | **내부 테스트용 가짜 WCS 서버** (표준 라이브러리만) — 오더를 로봇에 POST하고 콜백을 받음 |
| `tools/fake_wcs/sim_robot.py` | 로봇 없이 서버를 검증하는 가짜 로봇 |

## 통신 방향 (AMR Transport Order 규격 준용)

AMR 쪽 SFA-WCS 규격(v07.4)과 같은 형태로, **WCS가 로봇에 오더를 push**하고 로봇이 완료/실패를 콜백한다.
파지 전/배치 전에는 도착 보고 후 **PIO readiness(9장)를 GET으로 확인**해 READY일 때만 진행한다.

```
WCS ──POST /api/v1/wcs/transport-orders──▶ RBY1 오더 서버 (:5225)   → 201 {"orderStatus":"ACCEPTED"}
RBY1 ──POST /api/v1/rb/rby1/status (1Hz)──▶ WCS                       (기존 상태 업로드, work_cycle/error_message 포함)
RBY1 ──POST /api/v1/rb/transport-events──▶ WCS                        ARRIVED_AT_FROM / ARRIVED_AT_TO (도착 보고)
RBY1 ──GET /api/v1/rb/stations/{stationId}/readiness?operation=LOAD|UNLOAD──▶ WCS   READY / NOT_READY (2초마다 재확인)
RBY1 ──POST /api/v1/rb/transport-events──▶ WCS                        COMPLETED / FAILED / CANCELED (+message)
```

한 사이클 안의 순서: 오더 수신 → `ARRIVED_AT_FROM` → LOAD readiness READY 대기 → 파지·이송·AR 정렬 →
`ARRIVED_AT_TO` → UNLOAD readiness READY 대기 → 배치 → 복귀 후진(BACK) → **`COMPLETED`** → 회전·복귀 주행(오더 밖).
COMPLETED 이후 로봇은 수 초간 더 복귀 중이라 status의 work_cycle은 WORKING을 유지하다 사이클이 끝나면 DONE → IDLE이 된다.
그 구간의 실패는 오더가 이미 종료된 뒤라 FAILED 콜백 없이 status의 ERROR/error_message로만 전달된다.
readiness는 NOT_READY(또는 조회 실패)면 `READINESS_POLL_SEC`(2초)마다 재확인하고, `READINESS_MAX_WAIT_SEC`(120초)를 넘기면
`PIO_NOT_READY_TIMEOUT`을 message로 `FAILED`를 보고한다. 대기 중 취소 요청이 오면 `CANCELED`로 끝난다.

## 가짜 WCS로 테스트하기

### 1. 서버 기동 (아무 PC)

```bash
python tools/fake_wcs/server.py                                   # :5224, 로봇 http://127.0.0.1:5225 에 오더 발행 (수동)
FAKE_WCS_AUTO_DISPATCH=1 python tools/fake_wcs/server.py          # 자동 반복 발행 (완료 후 COOLDOWN 지나면 다음 오더)
FAKE_WCS_ROBOT_URL=http://<로봇PC IP>:5225 FAKE_WCS_COOLDOWN_SEC=15 python tools/fake_wcs/server.py
```

기본은 **수동 발행**이라 서버를 띄워도 오더가 나가지 않는다. 대시보드의 **오더 발행** 카드에서
wcsOrderId/carrierId(비우면 자동 채번), fromStationId/toStationId, priority를 넣고 **[오더 발행]**을 누르면 로봇에 POST된다.
같은 카드의 **자동 발행** 체크박스를 켜면 READY가 될 때마다 기본값 오더를 자동 발행한다(기존 동작).

대시보드: `http://localhost:5224` — 서버 상태/발행 모드/현재 오더/대기 잔여, 오더 발행 폼([오더 발행]·[현재 오더 취소]·[재개]),
**PIO Readiness** 카드(스테이션별 LOAD/UNLOAD를 READY/NOT_READY로 토글, 로봇이 폴링 중이면 "로봇 대기 중" 표시. 기본 READY),
오더·이벤트 이력, **통신 로그**(보낸 오더/취소 요청과 로봇 회신, 로봇이 보낸 COMPLETED/FAILED/CANCELED 이벤트와 WCS 응답을
요청→응답 JSON 원본을 status payload처럼 항상 펼쳐서 표시), 로봇 상태(작업·에러 메시지 포함), 배터리, pose, 엔코더 26관절, 원본 payload.
버튼은 상태에 맞을 때만 활성화된다: 발행은 READY/COOLDOWN(COOLDOWN이면 대기를 끊고 즉시 발행), 취소는 RUNNING, 재개는 HALTED.

### 2. 데모 실행 (로봇 PC)

```bash
WCS_BASE_URL=http://<서버PC IP>:5224 DRY_RUN=0 python demo_full_sequence_loop.py
```

- `DRY_RUN` 기본값이 **true**(`communication/wcs/config.py`)라 실제 통신하려면 반드시 `DRY_RUN=0`.
  DRY_RUN=1이면 오더 서버를 띄우지 않고 가짜 오더가 즉시 발급된 것처럼 동작해 기존과 같은 연속 반복.
- 로봇 PC의 **5225 포트**가 서버 PC에서 접근 가능해야 한다 (`ROBOT_ORDER_PORT`로 변경 가능).
- **WCS가 닫혀 있어도 데모는 그대로 시작된다.** 시작 시 health check 실패는 경고 1회만 남기고 진행하며,
  status는 1Hz로 계속 재시도하다가 서버가 열리면 다음 주기(≤1초)에 `status POST 복구`를 찍고 전송을 재개한다.
  전송하지 못한 완료/실패(transport-event)와 `work_cycle` 전환은 큐에 남아 연결되는 즉시 순서대로 전송된다
  (transport-event는 같은 `eventId` 유지). 큐는 **각각 최대 10건**이고 넘치면 가장 오래된 것부터 버리며 경고를 남긴다
  (`PENDING_EVENT_MAXLEN` / `WORK_EVENT_MAXLEN`). 로봇 상태 스냅샷은 최신 1건만 유지하므로 쌓이지 않는다.
- `--no-wcs-order`: 오더 대기 없이 연속 반복 (기존 동작). 도착 보고/readiness 확인도 하지 않는다.
- readiness 대기 시간은 `READINESS_POLL_SEC`(기본 2), `READINESS_MAX_WAIT_SEC`(기본 120)으로 조정한다.
- 로봇 2대 이상이면 `ROBOT_SERIAL=RBY1-002` 로 구분.

### 3. 로봇 없이 서버만 검증

```bash
FAKE_WCS_AUTO_DISPATCH=1 python tools/fake_wcs/server.py                # 자동 발행으로 서버 기동 (수동이면 대시보드에서 발행)
DRY_RUN=0 python tools/fake_wcs/sim_robot.py --work-sec 3 --cycles 2   # 2사이클 후 종료 (--cycles 0: 무한)
DRY_RUN=0 python tools/fake_wcs/sim_robot.py --error "테스트 오류"       # FAILED → 서버 HALTED → 대시보드 [재개]
```

`DRY_RUN=0`을 빼면(기본 true) 아무것도 보내지 않는다. sim_robot은 `requests`가 필요하므로 그 패키지가 있는 python으로 실행한다.
정상이면 가짜 로봇 로그가 아래 순서로 흐른다 (도착 보고와 readiness 조회는 데모와 같은 위치에서 나간다):

```
WCS health check: HTTP 200 -> OK                              로봇→서버 GET /health
반송 오더 수신 서버 시작: http://0.0.0.0:5225/...              로봇 측 오더 서버 오픈
WCS status POST OK -> HTTP 201                                 로봇→서버 상태 업로드 (이후 1Hz)
반송 오더 수신: WCS-20260903-000001 (CV02_OUT -> RACK01_PORT02)  서버→로봇 오더 POST
사이클 1 시작
WCS transport-event OK: ARRIVED_AT_FROM …                      From 도착 보고
LOAD readiness CV02_OUT: READY (1회 조회, 0초) -> 로딩 시작
WCS transport-event OK: ARRIVED_AT_TO …                        To 도착 보고
UNLOAD readiness RACK01_PORT02: READY (1회 조회, 0초) -> 언로딩 시작
사이클 1 완료
WCS transport-event OK: COMPLETED WCS-…-000001 -> HTTP 200     로봇→서버 완료 콜백
(서버가 COOLDOWN 후 오더 #2 발행 → 반복)
WCS publisher 종료: sent=N failed=0
```

서버 로그에는 `[READY -> RUNNING] 오더 발행 … ACCEPTED` → `transport-event 수신: COMPLETED` → `[RUNNING -> COOLDOWN]` 전이가 찍힌다.
로봇이 아직 안 떠 있을 때 서버가 "로봇 오더 서버에 연결할 수 없습니다 — 2초마다 재시도"를 내는 것은 정상이다.

COMPLETED는 복귀 후진이 끝나는 시점에 오므로, 그 뒤 WCS가 바로 다음 오더를 POST하면 로봇 오더 서버 큐에 쌓였다가
복귀가 끝난 뒤 시작된다(자동 발행 모드에서는 COOLDOWN이 그 여유 역할). COMPLETED 이후 취소 요청은 409 `ORDER_ALREADY_FINALIZED`.

readiness 시나리오(규격 11장 T-03/T-04) 재현:

- **T-03 NOT_READY 후 READY**: PIO 카드에서 `CV02_OUT` LOAD를 NOT_READY로 두고 [오더 발행] → 로봇이 파지 전에 멈춰
  "NOT_READY — 2초마다 재확인"을 남기고 카드에 "로봇 대기 중 · N회"가 뜬다 → READY를 누르면 즉시 진행.
- **T-04 대기 초과**: `READINESS_MAX_WAIT_SEC=10`으로 로봇을 띄우고 `RACK01_PORT02` UNLOAD를 NOT_READY로 두면 10초 후 로봇이
  `PIO_NOT_READY_TIMEOUT: UNLOAD RACK01_PORT02 10s`를 message로 FAILED를 보고하고 서버는 HALTED가 된다.
- **대기 중 취소**: NOT_READY로 로봇이 기다리는 동안 [현재 오더 취소] → 로봇이 다음 폴링 전에 CANCELED를 보고한다.

### 4. curl로 개별 엔드포인트 확인

```bash
curl localhost:5224/health                                          # healthy
curl localhost:5224/api/test/state | python3 -m json.tool            # 서버 상태·오더/이벤트 이력 JSON
curl localhost:5224/api/v1/rb/rby1/status/RBY1-001/latest            # 마지막 수신 status 레코드
curl -X POST localhost:5224/api/test/resume                          # HALTED 해제
curl -X POST localhost:5224/api/test/cancel                          # 진행 중 오더 취소 요청 (RUNNING일 때만)
# 수동 발행: 필드는 모두 선택, 비우면 기본값(wcsOrderId/carrierId 자동 채번, from/to는 FAKE_WCS_*_STATION, priority 5)
curl -X POST localhost:5224/api/test/order -H 'Content-Type: application/json' -d '{"toStationId":"CV03_IN","priority":7}'
# → 200 {"ok":true,"message":"발행 완료 …"} / RUNNING 등 발행 불가 상태면 409 / 로봇 연결 실패 502 / priority 정수 아님 400
curl -X POST localhost:5224/api/test/auto -H 'Content-Type: application/json' -d '{"enabled":true}'   # 자동 발행 토글
# PIO readiness (v07.4 9장): 로봇이 묻는 GET 과 설비 상태를 바꾸는 테스트 전용 POST
curl 'localhost:5224/api/v1/rb/stations/CV02_OUT/readiness?operation=LOAD&wcsOrderId=X'            # 기본 READY
curl -X POST localhost:5224/api/test/readiness -H 'Content-Type: application/json' \
  -d '{"stationId":"RACK01_PORT02","operation":"UNLOAD","status":"NOT_READY","reasonCode":"RACK_FULL"}'

# 로봇 오더 서버가 떠 있을 때(sim_robot 또는 데모 실행 중) 오더를 직접 POST
curl -X POST localhost:5225/api/v1/wcs/transport-orders -H 'Content-Type: application/json' \
  -d '{"wcsOrderId":"T-1","carrierId":"TOTE-1","fromStationId":"CV02_OUT","toStationId":"RACK01_PORT02","priority":5,"timestamp":"x"}'
# → 201 ACCEPTED. 같은 내용 재전송 → 200(멱등). toStationId를 바꿔 재전송 → 409 DUPLICATE_ORDER_CONFLICT
```

## 시퀀스

```
서버 기동 ─ READY ─(로봇 :5225 에 오더 POST, 연결 안 되면 2초마다 재시도)
  로봇 ACCEPTED ─ RUNNING
  로봇: IDLE → WORKING → (사이클) → DONE, COMPLETED 콜백 → IDLE
  서버: COMPLETED 수신 ─ COOLDOWN 15초 ─ READY ─ 다음 오더 POST … 반복
  로봇 FAILED 콜백(message 포함) ─ HALTED (발행 중단, 에러 표시) ─ [재개] ─ READY
  대시보드 [현재 오더 취소] ─ 로봇에 취소 POST(202) ─ CANCEL_REQUESTED ─ 로봇이 단계 경계에서
  정지 후 CANCELED 콜백 ─ COOLDOWN ─ 다음 오더
```

## 통신 규격 (테스트용 최소, AMR v07.2 축소판)

| 방향 | 메서드/경로 | 내용 |
|---|---|---|
| WCS→로봇 | `POST /api/v1/wcs/transport-orders` | `{wcsOrderId, carrierId, fromStationId, toStationId, priority, timestamp}` → 201 `{wcsOrderId, orderStatus:"ACCEPTED", timestamp}`. 동일 ID 재전송 200(멱등), 내용 다르면 409 `DUPLICATE_ORDER_CONFLICT`, 필수값 누락 400 |
| WCS→로봇 | `POST /api/v1/wcs/transport-orders/{wcsOrderId}/cancel` | `{reasonCode, reason?, requestedAt}` → 202 `CANCEL_REQUESTED`(접수) 후 로봇이 단계 경계에서 안전 정지하고 `CANCELED` 콜백. 중복 200, 미존재 404, 이미 종료 409 (v07.3 4.8 준용) |
| 로봇→WCS | `POST /api/v1/rb/transport-events` | `{eventId, wcsOrderId, eventType: ARRIVED_AT_FROM\|ARRIVED_AT_TO\|COMPLETED\|FAILED\|CANCELED, robotSerial, nodeId?, result, message, occurredAt}` → `{accepted:true, eventId, receivedAt}`. eventId로 멱등, 실패 시 큐에 보관 후 재시도. 도착 보고 2종은 오더 상태를 바꾸지 않는다. COMPLETED 시점 = 배치 후 복귀 후진(BACK) 완료 직후 |
| 로봇→WCS | `GET /api/v1/rb/stations/{stationId}/readiness?operation=LOAD\|UNLOAD&wcsOrderId=` | v07.4 9장 PIO readiness → `{stationId, operation, status: READY\|NOT_READY, ready, reasonCode, updatedAt}`. 파지 전 LOAD(fromStationId), 배치 전 UNLOAD(toStationId). NOT_READY/조회 실패면 2초마다 재확인, 120초 초과 시 FAILED(`PIO_NOT_READY_TIMEOUT`) |
| 로봇→WCS | `POST /api/v1/rb/rby1/status` | 기존 status payload 1Hz → 201 `{"accepted":true}` |
| 로봇→WCS | `GET /health` | `healthy` |
| 조회 | `GET /api/v1/rb/rby1/status/{serial}/latest`, `…/history?limit=N` | 실 WCS와 같은 평탄화 레코드 + `errorMessage` 컬럼 |
| 테스트 | `POST /api/test/order`, `POST /api/test/auto`, `POST /api/test/readiness`, `POST /api/test/cancel`, `POST /api/test/resume`, `GET /api/test/state` | 수동 발행 / 자동 발행 토글 / readiness 설정 / 취소 / HALTED 해제 / 대시보드용 상태 |

실 WCS와 다른 점: `/api/test/*`는 가짜 서버 전용. 실 WCS는 `error_message`를 컬럼으로 저장하지 않는다(2026-08-27 기준).
