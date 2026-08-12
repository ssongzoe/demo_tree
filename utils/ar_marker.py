#!/usr/bin/env python3
"""RealSense 기반 AR 마커 검출 유틸리티.

핵심 기능
- RealSenseCamera: 카메라 연결 및 intrinsic 제공
- detect_ar_markers(): 마커 위치 / 자세 계산
- measure_marker(): 여러 프레임의 중앙값으로 위치 / yaw 측정
"""

from dataclasses import dataclass, field
import time

import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs


ARUCO_DICTS = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_5X5_50": aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_5X5_250": aruco.DICT_5X5_250,
    "DICT_6X6_50": aruco.DICT_6X6_50,
    "DICT_6X6_100": aruco.DICT_6X6_100,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_7X7_50": aruco.DICT_7X7_50,
    "DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11,
    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
}


@dataclass
class ArMarker:
    id: int
    corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    rotation_matrix: np.ndarray = field(default=None)

    @property
    def position(self):
        return self.tvec

    @property
    def distance(self):
        return float(np.linalg.norm(self.tvec))


def build_detector_parameters():
    """AR 마커 검출 파라미터를 생성한다."""
    params = aruco.DetectorParameters()

    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01

    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 3
    params.adaptiveThreshConstant = 7

    params.polygonalApproxAccuracyRate = 0.01
    params.minDistanceToBorder = 3
    params.minMarkerPerimeterRate = 0.01
    params.perspectiveRemovePixelPerCell = 12

    return params


def create_detector(dict_name):
    """지정한 ArUco / AprilTag 딕셔너리의 detector를 생성한다."""
    if dict_name not in ARUCO_DICTS:
        raise ValueError(f"지원하지 않는 AR dictionary: {dict_name}")

    dictionary = aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    return aruco.ArucoDetector(dictionary, build_detector_parameters())


def marker_object_points(marker_size):
    """PnP 계산에 사용할 마커 코너 4점을 생성한다."""
    half = marker_size / 2.0

    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def detect_ar_markers(frame, detector, camera_matrix, dist_coeffs, marker_size):
    """영상에서 마커를 검출하고 카메라 좌표계 위치 / 자세를 계산한다."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)

    if ids is None:
        return []

    object_points = marker_object_points(marker_size)
    markers = []

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        image_points = marker_corners.reshape(4, 2).astype(np.float32)

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if not ok:
            continue

        rvec = rvec.flatten()
        tvec = tvec.flatten()
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        markers.append(
            ArMarker(
                id=int(marker_id),
                corners=image_points,
                rvec=rvec,
                tvec=tvec,
                rotation_matrix=rotation_matrix,
            )
        )

    return markers


def marker_plane_yaw_error_deg(marker):
    """마커 평면을 기준으로 모바일 베이스가 보정해야 할 yaw 오차 [deg]를 계산한다."""
    rotation = marker.rotation_matrix

    marker_up = rotation[:, 0]
    marker_up = marker_up / (np.linalg.norm(marker_up) + 1e-12)

    normal = rotation[:, 2]

    if normal[2] > 0:
        normal = -normal

    camera_back = np.array([0.0, 0.0, -1.0])

    normal_horizontal = normal - np.dot(normal, marker_up) * marker_up
    target_horizontal = camera_back - np.dot(camera_back, marker_up) * marker_up

    if np.linalg.norm(normal_horizontal) < 1e-6 or np.linalg.norm(target_horizontal) < 1e-6:
        return None

    error = np.arctan2(
        np.dot(marker_up, np.cross(target_horizontal, normal_horizontal)),
        np.dot(target_horizontal, normal_horizontal),
    )

    return float(np.degrees(error))


class RealSenseCamera:
    """RealSense 컬러 스트림과 카메라 intrinsic을 제공한다."""

    def __init__(self, width, height, fps, serial=None):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if serial:
            self.config.enable_device(serial)

        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.camera_matrix = None
        self.dist_coeffs = None

    def start(self):
        """카메라 스트림을 시작한다."""
        profile = self.pipeline.start(self.config)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()

        self.camera_matrix = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.dist_coeffs = np.array(intr.coeffs, dtype=np.float64)

    def read(self, timeout_ms=5000):
        """다음 컬러 프레임을 BGR numpy 배열로 반환한다."""
        frames = self.pipeline.wait_for_frames(timeout_ms)
        color = frames.get_color_frame()

        if not color:
            return None

        return np.asanyarray(color.get_data())

    def stop(self):
        """카메라 스트림을 종료한다."""
        try:
            self.pipeline.stop()
        except Exception:
            pass


def measure_marker(cam, detector, marker_id, marker_size, measure_frames=8, flush_frames=8, timeout_s=15.0):
    """여러 프레임의 중앙값으로 마커 위치와 yaw 오차를 측정한다."""
    positions = []
    yaw_errors = []

    flushed = 0
    start = time.monotonic()

    while len(positions) < measure_frames and time.monotonic() - start < timeout_s:
        frame = cam.read()

        if frame is None:
            continue

        if flushed < flush_frames:
            flushed += 1
            continue

        markers = detect_ar_markers(frame, detector, cam.camera_matrix, cam.dist_coeffs, marker_size)
        markers = [marker for marker in markers if marker.id == marker_id]

        if not markers:
            continue

        marker = min(markers, key=lambda item: item.distance)
        yaw_error = marker_plane_yaw_error_deg(marker)

        if yaw_error is None:
            continue

        positions.append(marker.position)
        yaw_errors.append(yaw_error)

    if len(positions) < max(3, measure_frames // 2):
        return None, None

    position = np.median(np.asarray(positions), axis=0)
    yaw_error = float(np.median(yaw_errors))

    return position, yaw_error
