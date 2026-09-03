# demo_tree — RB-Y1 tote 이송 데모 + SFA WCS 연동

## 구성

| 경로 | 역할 |
|---|---|
| `demo_full_sequence_loop.py` | tote 인식/파지 → 이송 → AR 정렬 → 배치 → 복귀 반복 데모. WCS 상태 업로드 + 반송 오더 수신/완료 보고 |
| `communication/wcs/` | SFA WCS 통신 모듈 (status POST 1Hz, 오더 수신 서버, transport-events 콜백) |
| `tools/fake_wcs/server.py` | **내부 테스트용 가짜 WCS 서버** (표준 라이브러리만) — 오더를 로봇에 POST하고 콜백을 받음 |
| `tools/fake_wcs/sim_robot.py` | 로봇 없이 서버를 검증하는 가짜 로봇 |

## 통신 방향 (AMR Transport Order 규격 준용)

AMR 쪽 SFA-WCS 규격(v07.2)과 같은 형태로, **WCS가 로봇에 오더를 push**하고 로봇이 완료/실패를 콜백한다.

```
WCS ──POST /api/v1/wcs/transport-orders──▶ RBY1 오더 서버 (:5225)   → 201 {"orderStatus":"ACCEPTED"}
RBY1 ──POST /api/v1/rb/rby1/status (1Hz)──▶ WCS                       (기존 상태 업로드, work_cycle/error_message 포함)
RBY1 ──POST /api/v1/rb/transport-events──▶ WCS                        COMPLETED / FAILED (+message)
```

## 가짜 WCS로 테스트하기

### 1. 서버 기동 (아무 PC)

```bash
python tools/fake_wcs/server.py                                   # :5224, 로봇 http://127.0.0.1:5225 에 오더 발행
FAKE_WCS_ROBOT_URL=http://<로봇PC IP>:5225 FAKE_WCS_COOLDOWN_SEC=15 python tools/fake_wcs/server.py
```

대시보드: `http://localhost:5224` — 서버 상태/현재 오더/대기 잔여, 오더·이벤트 이력, 로봇 상태(작업·에러 메시지 포함),
배터리, pose, 엔코더 26관절, 원본 payload. FAILED로 멈추면 **[재개]** 버튼.

### 2. 데모 실행 (로봇 PC)

```bash
WCS_BASE_URL=http://<서버PC IP>:5224 DRY_RUN=0 python demo_full_sequence_loop.py
```

- `DRY_RUN` 기본값이 **true**(`communication/wcs/config.py`)라 실제 통신하려면 반드시 `DRY_RUN=0`.
  DRY_RUN=1이면 오더 서버를 띄우지 않고 가짜 오더가 즉시 발급된 것처럼 동작해 기존과 같은 연속 반복.
- 로봇 PC의 **5225 포트**가 서버 PC에서 접근 가능해야 한다 (`ROBOT_ORDER_PORT`로 변경 가능).
- `--no-wcs-order`: 오더 대기 없이 연속 반복 (기존 동작).
- 로봇 2대 이상이면 `ROBOT_SERIAL=RBY1-002` 로 구분.

### 3. 로봇 없이 서버만 검증

```bash
DRY_RUN=0 python tools/fake_wcs/sim_robot.py --work-sec 3          # 무한 반복
DRY_RUN=0 python tools/fake_wcs/sim_robot.py --error "테스트 오류"   # FAILED → 서버 HALTED → 대시보드 [재개]
```

## 시퀀스

```
서버 기동 ─ READY ─(로봇 :5225 에 오더 POST, 연결 안 되면 2초마다 재시도)
  로봇 ACCEPTED ─ RUNNING
  로봇: IDLE → WORKING → (사이클) → DONE, COMPLETED 콜백 → IDLE
  서버: COMPLETED 수신 ─ COOLDOWN 15초 ─ READY ─ 다음 오더 POST … 반복
  로봇 FAILED 콜백(message 포함) ─ HALTED (발행 중단, 에러 표시) ─ [재개] ─ READY
```

## 통신 규격 (테스트용 최소, AMR v07.2 축소판)

| 방향 | 메서드/경로 | 내용 |
|---|---|---|
| WCS→로봇 | `POST /api/v1/wcs/transport-orders` | `{wcsOrderId, carrierId, fromStationId, toStationId, priority, timestamp}` → 201 `{wcsOrderId, orderStatus:"ACCEPTED", timestamp}`. 동일 ID 재전송 200(멱등), 내용 다르면 409 `DUPLICATE_ORDER_CONFLICT`, 필수값 누락 400 |
| 로봇→WCS | `POST /api/v1/rb/transport-events` | `{eventId, wcsOrderId, eventType: COMPLETED\|FAILED, robotSerial, result, message, occurredAt}` → `{accepted:true, eventId, receivedAt}`. eventId로 멱등, 로봇은 3회 재시도 |
| 로봇→WCS | `POST /api/v1/rb/rby1/status` | 기존 status payload 1Hz → 201 `{"accepted":true}` |
| 로봇→WCS | `GET /health` | `healthy` |
| 조회 | `GET /api/v1/rb/rby1/status/{serial}/latest`, `…/history?limit=N` | 실 WCS와 같은 평탄화 레코드 + `errorMessage` 컬럼 |
| 테스트 | `POST /api/test/resume`, `GET /api/test/state` | HALTED 해제 / 대시보드용 상태 |

실 WCS와 다른 점: `/api/test/*`는 가짜 서버 전용. 실 WCS는 `error_message`를 컬럼으로 저장하지 않는다(2026-08-27 기준).
오더 취소(AMR 4.8)는 아직 미구현.
