# demo_tree — RB-Y1 tote 이송 데모 + SFA WCS 연동

## 구성

| 경로 | 역할 |
|---|---|
| `demo_full_sequence_loop.py` | tote 인식/파지 → 이송 → AR 정렬 → 배치 → 복귀 반복 데모. WCS 상태 업로드 + START 명령 수신 |
| `communication/wcs/` | SFA WCS 통신 모듈 (status POST 1Hz, command GET 폴링) |
| `tools/fake_wcs/server.py` | **내부 테스트용 가짜 WCS 서버** (표준 라이브러리만) |
| `tools/fake_wcs/sim_robot.py` | 로봇 없이 서버를 검증하는 가짜 로봇 |

## 가짜 WCS로 테스트하기

### 1. 서버 기동 (아무 PC)

```bash
python tools/fake_wcs/server.py                      # 0.0.0.0:5224, 완료 후 대기 15초
FAKE_WCS_COOLDOWN_SEC=5 python tools/fake_wcs/server.py
```

대시보드: `http://localhost:5224` — 서버 상태/현재 명령/대기 잔여, 로봇 상태(작업·에러 메시지 포함),
배터리, pose, 엔코더 26관절, 원본 payload, 명령 이력. ERROR로 멈추면 **[재개]** 버튼.

### 2. 데모 실행 (로봇 PC)

```bash
WCS_BASE_URL=http://<서버PC IP>:5224 DRY_RUN=0 python demo_full_sequence_loop.py
```

- `DRY_RUN` 기본값이 **true**(`communication/wcs/config.py`)라 실제 전송하려면 반드시 `DRY_RUN=0`.
  DRY_RUN=1이면 서버 없이 START가 즉시 발급된 것처럼 동작해 기존과 같은 연속 반복.
- `--no-wcs-command`: 명령 대기 없이 연속 반복 (기존 동작).
- 로봇 2대 이상이면 `ROBOT_SERIAL=RBY1-002` 로 구분.

### 3. 로봇 없이 서버만 검증

```bash
python tools/fake_wcs/sim_robot.py --work-sec 3          # 무한 반복
python tools/fake_wcs/sim_robot.py --error "테스트 오류"   # ERROR → 서버 HALTED → 대시보드 [재개]
```

## 시퀀스

```
서버 기동 ─ READY
  로봇 폴링 ──> START(commandId=1) 발행 ─ RUNNING
  로봇: IDLE → WORKING → (사이클) → DONE → IDLE
  서버: WORKING 확인 후 DONE 수신 ─ COOLDOWN 15초
  만료 후 로봇 폴링 ──> START(commandId=2) … 반복
  로봇 ERROR 수신 ─ HALTED (명령 WAIT 고정, error_message 표시) ─ [재개] ─ READY
```

## 통신 규격 (테스트용 최소)

| 방향 | 메서드/경로 | 내용 |
|---|---|---|
| 로봇→서버 | `GET /health` | `healthy` |
| 로봇→서버 | `POST /api/v1/rb/rby1/status` | 기존 status payload (`robotSerial`, `robot_state.work_cycle/error_message`, `encoder`, `pose`, …) → 201 `{"accepted":true}` |
| 로봇→서버 | `GET /api/v1/rb/rby1/command/{serial}` | `{"commandId": n, "command": "START"\|"WAIT", "issuedAt": "…Z"}` — 1Hz 폴링, 새 commandId의 START만 소비 |
| 조회 | `GET /api/v1/rb/rby1/status/{serial}/latest`, `…/history?limit=N` | 실 WCS와 같은 평탄화 레코드 + `errorMessage` 컬럼 |
| 테스트 | `POST /api/test/resume`, `GET /api/test/state` | HALTED 해제 / 대시보드용 상태 |

실 WCS와 다른 점: 명령 엔드포인트와 `/api/test/*`는 가짜 서버 전용이며, 실 WCS는 `error_message`를 컬럼으로 저장하지 않는다(2026-08-27 기준).
