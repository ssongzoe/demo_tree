# RB-Y1 Demo Tree

RB-Y1의 모바일 주행, 로봇 제어, 비전 기반 정렬 기능을 재사용 가능한 형태로 구성한 데모 프로젝트입니다.

## Project Structure

```text
demo_tree/
├── control/
│   ├── __init__.py
│   ├── mobile_controller.py
│   └── robot_controller.py
│
├── skills/
│   └── ar_align.py
│
├── utils/
│   └── ar_marker.py
│
├── demo_mobile_turn_and_go.py
├── demo_go_and_align.py
└── README.md
```

### `control/`

RB-Y1 SDK를 이용한 로봇의 기본 제어 기능을 담당합니다.

* `mobile_controller.py`

  * 모바일 베이스 초기화
  * Odometry 수신
  * SE(2) 기반 전후 / 좌우 / 회전 주행
  * Trajectory 및 feedback 제어

* `robot_controller.py`

  * Torso / 양팔 / Head 제어
  * Joint Position 기반 자세 제어
  * READY Pose 등 공통 자세 관리

### `skills/`

데모에서 반복적으로 사용할 수 있는 작업 단위 기능을 정의합니다.

* `ar_align.py`

  * AR 마커 기반 모바일 베이스 정렬
  * 마커 yaw 정렬
  * 마커 기준 전후 / 좌우 위치 정렬

### `utils/`

센서 및 인식과 관련된 공통 기능을 담당합니다.

* `ar_marker.py`

  * RealSense 카메라 연결
  * ArUco / AprilTag 검출
  * 마커의 3D 위치 및 자세 계산
  * 여러 프레임을 이용한 안정적인 마커 측정

### Demo

* `demo_mobile_turn_and_go.py`

  * 회전하며 이동 후 직진하는 기본 모바일 주행 데모

* `demo_go_and_align.py`

  * 0.80 m 직진 후 AR 마커를 인식하여 위치와 방향을 정렬하는 데모

## 구조

```text
Demo
 ├── Control   → 로봇을 어떻게 움직일지
 ├── Skills    → 어떤 작업을 수행할지
 └── Utils     → 센서 정보를 어떻게 얻을지
```

각 데모에서는 필요한 Control과 Skill을 조합하여 새로운 작업 시나리오를 구성합니다.
