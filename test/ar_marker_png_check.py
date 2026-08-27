#!/usr/bin/env python3
"""PNG 이미지에서 AprilTag 검출 상태를 확인한다.

- marker ID를 몰라도 사용 가능
- 검출된 모든 ID와 corner 정보를 출력
- --marker-id를 주지 않으면 검출된 마커가 하나일 때 자동 선택
- 여러 개가 검출되면 영상에서 가장 크게 보이는 마커를 자동 선택
"""


import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent


for path in (PROJECT_ROOT, TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.ar_marker import create_detector

ARUCO_DICT = "DICT_APRILTAG_36h11"


def polygon_area(points: np.ndarray) -> float:
    return float(abs(cv2.contourArea(points.astype(np.float32))))


def marker_metrics(corners: np.ndarray) -> dict:
    pts = corners.reshape(4, 2).astype(np.float64)

    edge_lengths = np.asarray(
        [
            np.linalg.norm(pts[1] - pts[0]),
            np.linalg.norm(pts[2] - pts[1]),
            np.linalg.norm(pts[3] - pts[2]),
            np.linalg.norm(pts[0] - pts[3]),
        ],
        dtype=np.float64,
    )

    center = pts.mean(axis=0)
    min_edge = float(edge_lengths.min())
    max_edge = float(edge_lengths.max())

    return {
        "corners": pts,
        "center": center,
        "edge_lengths": edge_lengths,
        "mean_edge": float(edge_lengths.mean()),
        "perspective_ratio": min_edge / max_edge if max_edge > 0.0 else 0.0,
        "area": polygon_area(pts),
    }


def check_marker_image(image_path: str, marker_id: int | None = None, output_path: str | None = None, show: bool = False) -> bool:
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"이미지를 읽지 못했습니다: {image_path}")

    detector = create_detector(ARUCO_DICT)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    print(f"이미지: {image_path}")
    print(f"해상도: {image.shape[1]}x{image.shape[0]}")
    print(f"Rejected candidates: {len(rejected)}")

    output = image.copy()

    if ids is None:
        print("검출된 AprilTag가 없습니다.")
        _save_or_show(output, image_path, output_path, show)
        return False

    ids_flat = ids.flatten()
    cv2.aruco.drawDetectedMarkers(output, corners, ids)
    print(f"검출 ID: {ids_flat.tolist()}")

    if marker_id is None:
        best_index = max(range(len(corners)), key=lambda idx: polygon_area(corners[idx].reshape(4, 2)))
        marker_id = int(ids_flat[best_index])

        if len(ids_flat) == 1:
            print(f"marker ID 자동 선택: {marker_id}")
        else:
            print(f"여러 marker가 검출되어 가장 크게 보이는 ID {marker_id}를 선택합니다.")
    else:
        target_indices = np.where(ids_flat == marker_id)[0]

        if len(target_indices) == 0:
            print(f"지정한 marker id={marker_id}는 검출되지 않았습니다.")
            _save_or_show(output, image_path, output_path, show)
            return False

        best_index = max(target_indices, key=lambda idx: polygon_area(corners[idx].reshape(4, 2)))

    metrics = marker_metrics(corners[best_index])

    print(f"\nmarker id={marker_id} 검출 성공")
    print(f"center            : ({metrics['center'][0]:.1f}, {metrics['center'][1]:.1f}) px")
    print(f"corners           : {np.round(metrics['corners'], 1).tolist()}")
    print(f"edge lengths      : {np.round(metrics['edge_lengths'], 1).tolist()} px")
    print(f"mean edge         : {metrics['mean_edge']:.1f} px")
    print(f"area              : {metrics['area']:.0f} px^2")
    print(f"perspective ratio : {metrics['perspective_ratio']:.3f}  (1.0에 가까울수록 정면에 가까움)")

    if metrics["mean_edge"] < 25.0:
        print("판정: 검출은 됐지만 마커가 작습니다. 실제 주행 중 검출 안정성을 추가 확인하는 것이 좋습니다.")
    elif metrics["perspective_ratio"] < 0.45:
        print("판정: 검출은 됐지만 원근 왜곡이 큽니다. 헤드캠 각도 변화에서 검출 안정성을 추가 확인하는 것이 좋습니다.")
    else:
        print("판정: 이 이미지에서는 marker 검출 상태가 충분히 좋아 보입니다.")

    center = tuple(np.round(metrics["center"]).astype(int))
    cv2.circle(output, center, 5, (0, 0, 255), -1)
    cv2.putText(output, f"ID {marker_id}", (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    _save_or_show(output, image_path, output_path, show)
    return True


def _save_or_show(image: np.ndarray, input_path: str, output_path: str | None, show: bool) -> None:
    if output_path is None:
        src = Path(input_path)
        output_path = str(src.with_name(f"{src.stem}_detected.png"))

    cv2.imwrite(output_path, image)
    print(f"검출 결과 저장: {output_path}")

    if show:
        cv2.imshow("AR Marker PNG Check", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="PNG 이미지에서 AprilTag 검출 상태 확인")
    parser.add_argument("image", help="RealSense Viewer에서 저장한 PNG/JPG 경로")
    parser.add_argument("--marker-id", type=int, default=None, help="특정 ID만 확인할 때 사용, 생략하면 자동 선택")
    parser.add_argument("--output", default=None, help="검출 결과 이미지 저장 경로")
    parser.add_argument("--show", action="store_true", help="검출 결과 OpenCV 창 표시")
    args = parser.parse_args()

    success = check_marker_image(args.image, marker_id=args.marker_id, output_path=args.output, show=args.show)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
