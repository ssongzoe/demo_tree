"""가짜 WCS 서버 검증용 가짜 로봇 (실제 로봇/SDK 없이 통신 시퀀스만 재현).

    DRY_RUN=0 python tools/fake_wcs/sim_robot.py                        # 서버 localhost:5224, 오더 수신 :5225
    DRY_RUN=0 WCS_BASE_URL=http://localhost:5224 python tools/fake_wcs/sim_robot.py --work-sec 3 --cycles 2
    DRY_RUN=0 python tools/fake_wcs/sim_robot.py --error "테스트 오류"    # 첫 오더를 FAILED로 끝내고 종료

데모(demo_full_sequence_loop.py)와 같은 통신 모듈(communication.wcs)을 그대로 쓴다:
IDLE(하트비트) → WCS 반송 오더 수신(POST :5225) → WORKING(work-sec) → DONE + COMPLETED 콜백 → IDLE … 반복.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import threading
import time
from types import SimpleNamespace

# 리포 루트를 import 경로에 추가 (tools/fake_wcs/ 에서 실행해도 communication.wcs를 찾도록)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from communication.wcs.publisher import WcsPublisher  # noqa: E402

# Model M 26 DOF 파트 인덱스 (rby1-sdk Model_M 순서)
_MODEL = SimpleNamespace(
    robot_joint_names=[f"j{i}" for i in range(26)],
    mobility_idx=[0, 1, 2, 3],
    torso_idx=[4, 5, 6, 7, 8, 9],
    right_arm_idx=list(range(10, 17)),
    left_arm_idx=list(range(17, 24)),
    head_idx=[24, 25],
)


def _fake_state(t: float):
    """state.py의 _to_snapshot이 읽는 속성만 갖춘 가짜 RobotState."""
    on = SimpleNamespace(name="PowerOn")
    return SimpleNamespace(
        is_ready=[True] * 26,
        emo_states=[SimpleNamespace(state=SimpleNamespace(name="Released"))],
        power_states=[SimpleNamespace(state=on)],
        joint_states=[SimpleNamespace(power_on=True)] * 26,
        battery_state=SimpleNamespace(level_percent=87.0 - 0.01 * t, voltage=51.8, current=-3.2),
        position=[0.3 * math.sin(0.2 * t + i * 0.5) for i in range(26)],
        odometry=[0.5 * math.sin(0.05 * t), 0.5 * math.cos(0.05 * t), (0.1 * t) % 6.283],
        system_stat=SimpleNamespace(cpu_usage=21.5, memory_usage=43.0, uptime=t),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="가짜 WCS 검증용 가짜 로봇")
    parser.add_argument("--work-sec", type=float, default=3.0, help="WORKING 유지 시간")
    parser.add_argument("--cycles", type=int, default=0, help="반복 횟수, 0이면 무한")
    parser.add_argument("--error", default=None, help="첫 오더를 이 메시지의 FAILED로 끝내고 종료")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("sim-robot")

    dry_run = os.getenv("DRY_RUN", "0").strip().lower() in ("1", "true", "yes", "y", "on")
    publisher = WcsPublisher(robot_model=_MODEL, dry_run=dry_run)

    t0 = time.monotonic()
    stop = threading.Event()

    def feed_state():  # SDK 콜백 대신 10Hz로 가짜 state 주입
        while not stop.is_set():
            publisher.on_state(_fake_state(time.monotonic() - t0), None)
            stop.wait(0.1)

    threading.Thread(target=feed_state, daemon=True).start()

    completed = 0
    try:
        publisher.set_work_state("IDLE")
        publisher.start()
        cycle = 1
        while args.cycles == 0 or cycle <= args.cycles:
            publisher.set_work_state("IDLE")
            log.info("WCS 반송 오더 대기 중...")
            order = publisher.wait_for_order()
            log.info("사이클 %d 시작: %s (%s -> %s)", cycle, order["wcsOrderId"],
                     order.get("fromStationId"), order.get("toStationId"))
            publisher.set_work_state("WORKING")
            time.sleep(args.work_sec)
            if args.error:
                raise RuntimeError(args.error)
            publisher.set_work_state("DONE")
            publisher.complete_order()
            log.info("사이클 %d 완료", cycle)
            completed += 1
            cycle += 1
            time.sleep(1.0)  # 데모와 동일: DONE이 최소 1회 업로드되도록
        publisher.set_work_state("IDLE")
    except KeyboardInterrupt:
        publisher.set_work_state("IDLE")
        publisher.fail_order("사용자가 데모를 중단했습니다")
        log.info("중단. 완료 사이클 %d", completed)
    except Exception as error:  # noqa: BLE001 - 데모와 같은 방식으로 FAILED 보고
        publisher.set_work_state("ERROR", str(error))
        publisher.fail_order(str(error))
        log.info("FAILED 보고: %s", error)
    finally:
        stop.set()
        publisher.stop(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
