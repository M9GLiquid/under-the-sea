"""Minimal GPS mapping API.

Provides a tiny, dependency-light API to map points from the original fisheye
image space (as provided by the GPS server) to the rectified canvas space used
for grids, and to compute grid cells.

Usage example:

    from src.tools.gps_api import GPSMapper

    mapper = GPSMapper(
        fisheye_json="data/GPS-Real_fisheye_calibration.json",
        transform_json="data/GPS-Real_corrected_transform.json",
    )

    xr, yr = mapper.map_original_to_rectified(258, 50)
    cell = mapper.grid_cell_from_original(258, 50, cols=11, rows=8)
    cell2 = mapper.grid_cell_from_rectified(xr, yr, cols=11, rows=8)
"""

from __future__ import annotations

import json
from typing import Dict, Tuple

import cv2
import numpy as np


# The original GPS server image size (width, height)
SERVER_SIZE: Tuple[int, int] = (2048, 1536)

# Must match fix_fisheye correction step; used to show more area in corrected image
SCALE_FACTOR: float = 0.8


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_new_camera_matrix(camera_matrix: np.ndarray, margin: int, scale_factor: float) -> np.ndarray:
    new_camera_matrix = camera_matrix.copy().astype(np.float64)
    new_camera_matrix[0, 2] += float(margin)
    new_camera_matrix[1, 2] += float(margin)
    new_camera_matrix[0, 0] *= float(scale_factor)
    new_camera_matrix[1, 1] *= float(scale_factor)
    return new_camera_matrix


def _apply_homography(x: float, y: float, homography: np.ndarray) -> Tuple[float, float]:
    v = np.array([x, y, 1.0], dtype=np.float64)
    q = homography @ v
    return float(q[0] / q[2]), float(q[1] / q[2])


class GPSMapper:
    """Small mapping helper with cached calibration and transform data."""

    def __init__(self, fisheye_json: str, transform_json: str) -> None:
        # Load fisheye calibration
        fisheye_bundle = _load_json(fisheye_json)
        fisheye = fisheye_bundle.get("fisheye_calibration", fisheye_bundle)
        self.camera_matrix = np.array(fisheye["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(fisheye["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)
        self.calib_size: Tuple[int, int] = (int(fisheye["image_size"][0]), int(fisheye["image_size"][1]))
        self.margin_pixels = int(fisheye.get("margin_pixels", 0))
        self.new_camera_matrix = _build_new_camera_matrix(self.camera_matrix, self.margin_pixels, SCALE_FACTOR)

        # Load rectification transform
        t = _load_json(transform_json)
        self.homography_image_to_world_canvas = np.array(t["homography_image_to_world_canvas"], dtype=np.float64)

        out_w = int(t["output_size"]["width"])
        out_h = int(t["output_size"]["height"])
        min_x = float(t["warp_bounds"]["min_x"])
        min_y = float(t["warp_bounds"]["min_y"])

        # Arena bounds (canvas coordinates)
        self.left: float = -min_x
        self.top: float = -min_y
        self.right: float = self.left + (out_w - 1)
        self.bottom: float = self.top + (out_h - 1)

    def map_original_to_rectified(self, x: float, y: float) -> Tuple[float, float]:
        """Map a point from original (GPS server) to rectified canvas pixels."""
        # Scale original point to the calibration resolution if needed
        sx = float(self.calib_size[0]) / float(SERVER_SIZE[0])
        sy = float(self.calib_size[1]) / float(SERVER_SIZE[1])
        x_cal = float(x) * sx
        y_cal = float(y) * sy

        # Undistort to corrected pixels
        pts = np.array([[[x_cal, y_cal]]], dtype=np.float64)
        undist = cv2.fisheye.undistortPoints(
            pts,
            self.camera_matrix,
            self.dist_coeffs,
            R=np.eye(3),
            P=self.new_camera_matrix,
        )
        x_corr = float(undist[0, 0, 0])
        y_corr = float(undist[0, 0, 1])

        # Rectify (corrected → canvas)
        return _apply_homography(x_corr, y_corr, self.homography_image_to_world_canvas)

    def grid_cell_from_rectified(self, x_rect: float, y_rect: float, cols: int, rows: int) -> Dict[str, object]:
        width = self.right - self.left
        height = self.bottom - self.top
        cell_w = max(1e-6, width / float(cols))
        cell_h = max(1e-6, height / float(rows))
        col = int((x_rect - self.left) // cell_w)
        row = int((y_rect - self.top) // cell_h)
        in_bounds = 0 <= col < cols and 0 <= row < rows
        return {
            "col": col,
            "row": row,
            "in_bounds": in_bounds,
            "cell_size_px": {"x": cell_w, "y": cell_h},
            "arena_bounds": {"left": self.left, "top": self.top, "right": self.right, "bottom": self.bottom},
        }

    def grid_cell_from_original(self, x: float, y: float, cols: int, rows: int) -> Dict[str, object]:
        xr, yr = self.map_original_to_rectified(x, y)
        info = self.grid_cell_from_rectified(xr, yr, cols, rows)
        info["rectified_xy"] = {"x": xr, "y": yr}
        info["original_xy"] = {"x": x, "y": y}
        return info
