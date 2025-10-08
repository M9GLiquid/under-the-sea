#!/usr/bin/env python3
"""
Tool #7 - Point Mapper (Original → Rectified)

Purpose:
- Show the original image (what the GPS server sees) and the rectified top-down
  image side-by-side.
- When you click a point on the original, the tool maps it through the saved
  fisheye calibration and rectification transform to the rectified image and
  marks it there.
- It also overlays the transformed original image bounds onto the rectified view
  so you can see how the 2048x1536 frame maps.

Controls:
- Left click (on Original window): add a point and map it to Rectified
- r: clear all points
- s: save a side-by-side snapshot to output/
- q or window close: quit

Data inputs (already in repo):
- data/GPS-Real_fisheye_calibration.json (camera intrinsics, distortion, margin)
- data/GPS-Real_corrected_transform.json (homographies and canvas translation)

Notes:
- If your GPS server coordinates are based on a specific original resolution
  (e.g., 2048x1536), this tool scales those coordinates into the calibration
  resolution recorded in the fisheye calibration JSON before undistortion.
"""

import os
import json
import time
import argparse
from typing import List, Tuple

import cv2
import numpy as np


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


# Window titles as constants to satisfy linter duplication warning
ORIGINAL_WIN_TITLE = "Original (Tool #7)"
RECTIFIED_WIN_TITLE = "Rectified (Tool #7)"


def _to_abs(path: str, project_root: str) -> str:
    # Normalize backslashes in JSON to current OS separators
    norm = path.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(norm):
        return norm
    return os.path.join(project_root, norm)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_output_dir(project_root: str) -> str:
    out_dir = os.path.join(project_root, "output")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _build_new_camera_matrix(camera_matrix: np.ndarray, margin: int, scale_factor: float) -> np.ndarray:
    """Replicate fix_fisheye's new camera matrix adjustments.
    - Shift principal point by margin
    - Scale focal length to show more area
    """
    new_camera_matrix = camera_matrix.copy().astype(np.float64)
    new_camera_matrix[0, 2] += float(margin)
    new_camera_matrix[1, 2] += float(margin)
    new_camera_matrix[0, 0] *= float(scale_factor)
    new_camera_matrix[1, 1] *= float(scale_factor)
    return new_camera_matrix


def _undistort_point_to_corrected(
    pt_xy: Tuple[float, float],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    new_camera_matrix: np.ndarray,
    server_size: Tuple[int, int],
    calib_size: Tuple[int, int],
) -> Tuple[float, float]:
    """Map a point from the original server image to the corrected image space.

    If the server image resolution differs from the calibration image_size, scale
    the input point into calibration space before undistortion so that K, D are
    applied consistently.
    """
    sx = float(calib_size[0]) / float(server_size[0])
    sy = float(calib_size[1]) / float(server_size[1])
    x_cal = float(pt_xy[0]) * sx
    y_cal = float(pt_xy[1]) * sy

    pts = np.array([[[x_cal, y_cal]]], dtype=np.float64)  # shape (1,1,2)
    undist = cv2.fisheye.undistortPoints(pts, camera_matrix, dist_coeffs, R=np.eye(3), P=new_camera_matrix)
    x_corr = float(undist[0, 0, 0])
    y_corr = float(undist[0, 0, 1])
    return x_corr, y_corr


def _apply_homography(pt_xy: Tuple[float, float], homography: np.ndarray) -> Tuple[float, float]:
    v = np.array([float(pt_xy[0]), float(pt_xy[1]), 1.0], dtype=np.float64)
    q = homography @ v
    if q[2] == 0:
        return float("nan"), float("nan")
    return float(q[0] / q[2]), float(q[1] / q[2])


def _sample_original_bounds(
    width: int,
    height: int,
    samples_per_edge: int = 128,
) -> List[Tuple[float, float]]:
    """Return dense polyline around the original (0,0)-(w-1,h-1) bounds."""
    w = float(width - 1)
    h = float(height - 1)
    pts: List[Tuple[float, float]] = []
    # Top edge left→right
    for i in range(samples_per_edge):
        t = i / float(samples_per_edge - 1)
        pts.append((t * w, 0.0))
    # Right edge top→bottom
    for i in range(1, samples_per_edge):
        t = i / float(samples_per_edge - 1)
        pts.append((w, t * h))
    # Bottom edge right→left
    for i in range(1, samples_per_edge):
        t = i / float(samples_per_edge - 1)
        pts.append(((1.0 - t) * w, h))
    # Left edge bottom→top
    for i in range(1, samples_per_edge - 1):
        t = i / float(samples_per_edge - 1)
        pts.append((0.0, (1.0 - t) * h))
    return pts


class PointMapperApp:
    def __init__(self, fisheye_json: str, transform_json: str, server_w: int, server_h: int, scale_factor: float = 0.8):
        self.project_root = _project_root()
        self.output_dir = _ensure_output_dir(self.project_root)

        # Load calibration
        self.fisheye_path = _to_abs(fisheye_json, self.project_root)
        self.transform_path = _to_abs(transform_json, self.project_root)
        self.fisheye = _load_json(self.fisheye_path)["fisheye_calibration"]
        self.transform = _load_json(self.transform_path)

        # Paths to images
        self.original_path = _to_abs(_load_json(self.fisheye_path).get("original_image", "images/GPS-Real.png"), self.project_root)
        self.rectified_path = _to_abs(self.transform.get("rectified_image", "output/GPS-Real_corrected_rectified_oriented.png"), self.project_root)

        # Load images
        self.img_original = cv2.imread(self.original_path)
        if self.img_original is None:
            raise FileNotFoundError(f"Could not load original image: {self.original_path}")
        self.img_rectified = cv2.imread(self.rectified_path)
        if self.img_rectified is None:
            raise FileNotFoundError(f"Could not load rectified image: {self.rectified_path}")

        # Server (GPS) image size and calibration image size
        self.server_size = (int(server_w), int(server_h))
        calib_size = self.fisheye.get("image_size", [self.img_original.shape[1], self.img_original.shape[0]])
        self.calib_size = (int(calib_size[0]), int(calib_size[1]))

        # Camera matrices
        camera_matrix = np.array(self.fisheye["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.array(self.fisheye["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)
        margin = int(self.fisheye.get("margin_pixels", 0))
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.new_camera_matrix = _build_new_camera_matrix(camera_matrix, margin, float(scale_factor))

        # Homographies
        self.homography_image_to_world_canvas = np.array(self.transform["homography_image_to_world_canvas"], dtype=np.float64)

        # Drawing state
        self.left_points: List[Tuple[int, int]] = []
        self.right_points: List[Tuple[int, int]] = []

        # Precompute transformed original bounds polyline on rectified
        self.bounds_poly = self._compute_bounds_polyline()

    def _compute_bounds_polyline(self) -> List[Tuple[int, int]]:
        w, h = self.server_size
        dense = _sample_original_bounds(w, h, samples_per_edge=128)
        mapped: List[Tuple[int, int]] = []
        for (x, y) in dense:
            xc, yc = _undistort_point_to_corrected((x, y), self.camera_matrix, self.dist_coeffs, self.new_camera_matrix, self.server_size, self.calib_size)
            xr, yr = _apply_homography((xc, yc), self.homography_image_to_world_canvas)
            mapped.append((int(round(xr)), int(round(yr))))
        return mapped

    def _draw_bounds_poly(self, img: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255)) -> None:
        if not self.bounds_poly:
            return
        pts = np.array(self.bounds_poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

    def _map_point(self, x: int, y: int) -> Tuple[int, int]:
        xc, yc = _undistort_point_to_corrected((x, y), self.camera_matrix, self.dist_coeffs, self.new_camera_matrix, self.server_size, self.calib_size)
        xr, yr = _apply_homography((xc, yc), self.homography_image_to_world_canvas)
        return int(round(xr)), int(round(yr))

    def _refresh_views(self) -> None:
        left = self.img_original.copy()
        right = self.img_rectified.copy()

        # Draw bounds on rectified
        self._draw_bounds_poly(right, (0, 255, 255))  # yellow

        # Draw points
        for i, (lx, ly) in enumerate(self.left_points):
            cv2.circle(left, (lx, ly), 4, (0, 0, 255), -1, cv2.LINE_AA)  # red
            label = f"P{i+1}"
            cv2.putText(left, label, (lx + 6, ly - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        for i, (rx, ry) in enumerate(self.right_points):
            cv2.circle(right, (rx, ry), 4, (0, 0, 255), -1, cv2.LINE_AA)
            label = f"P{i+1}"
            cv2.putText(right, label, (rx + 6, ry - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw crosshairs for consistent cursor visuals
        def draw_crosshair(img: np.ndarray, x: int, y: int) -> None:
            h, w = img.shape[:2]
            x = max(0, min(w - 1, int(x)))
            y = max(0, min(h - 1, int(y)))
            size = 10
            color = (255, 255, 255)
            thickness = 1
            cv2.line(img, (max(0, x - size), y), (min(w - 1, x + size), y), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x, max(0, y - size)), (x, min(h - 1, y + size)), color, thickness, cv2.LINE_AA)

        if not hasattr(self, "_mouse_left"):  # default positions
            self._mouse_left = [left.shape[1] // 2, left.shape[0] // 2]
            self._mouse_right = [right.shape[1] // 2, right.shape[0] // 2]
        draw_crosshair(left, self._mouse_left[0], self._mouse_left[1])
        draw_crosshair(right, self._mouse_right[0], self._mouse_right[1])

        cv2.imshow(ORIGINAL_WIN_TITLE, left)
        cv2.imshow(RECTIFIED_WIN_TITLE, right)

    def _on_left_click(self, x: int, y: int) -> None:
        xr, yr = self._map_point(x, y)
        self.left_points.append((x, y))
        self.right_points.append((xr, yr))
        print(f"Original → Rectified : ({x}, {y}) → ({xr}, {yr})")
        self._refresh_views()

    def _mouse_callback_original(self, event, x, y, flags, param):  # noqa: ANN001 - OpenCV signature
        if event == cv2.EVENT_LBUTTONDOWN:
            self._on_left_click(int(x), int(y))
        elif event == cv2.EVENT_MOUSEMOVE:
            self._mouse_left = [int(x), int(y)]
            self._refresh_views()

    def _mouse_callback_rectified(self, event, x, y, flags, param):  # noqa: ANN001 - OpenCV signature
        if event == cv2.EVENT_MOUSEMOVE:
            self._mouse_right = [int(x), int(y)]
            self._refresh_views()

    def run(self) -> int:
        cv2.namedWindow(ORIGINAL_WIN_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow(RECTIFIED_WIN_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        # Reasonable initial sizes
        h0, w0 = self.img_original.shape[:2]
        hr, wr = self.img_rectified.shape[:2]
        cv2.resizeWindow(ORIGINAL_WIN_TITLE, min(1200, w0), min(800, h0))
        cv2.resizeWindow(RECTIFIED_WIN_TITLE, min(1200, wr), min(800, hr))

        cv2.setMouseCallback(ORIGINAL_WIN_TITLE, self._mouse_callback_original)
        cv2.setMouseCallback(RECTIFIED_WIN_TITLE, self._mouse_callback_rectified)
        self._refresh_views()

        print("Point Mapper (Tool #7)")
        print("- Click on the Original window to map a point to Rectified")
        print("- Keys: r=reset, s=save side-by-side, q=quit")

        while True:
            # Exit if either window is closed
            if cv2.getWindowProperty(ORIGINAL_WIN_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.getWindowProperty(RECTIFIED_WIN_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKey(16) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.left_points.clear()
                self.right_points.clear()
                self._refresh_views()
            elif key == ord('s'):
                # Save side-by-side with overlays
                left = self.img_original.copy()
                right = self.img_rectified.copy()
                self._draw_bounds_poly(right, (0, 255, 255))
                for i, (lx, ly) in enumerate(self.left_points):
                    cv2.circle(left, (lx, ly), 4, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(left, f"P{i+1}", (lx + 6, ly - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                for i, (rx, ry) in enumerate(self.right_points):
                    cv2.circle(right, (rx, ry), 4, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(right, f"P{i+1}", (rx + 6, ry - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                h1 = left.shape[0]
                h2 = right.shape[0]
                H = max(h1, h2)
                # pad to same height
                if h1 != H:
                    pad = H - h1
                    left = cv2.copyMakeBorder(left, 0, pad, 0, 0, cv2.BORDER_CONSTANT)
                if h2 != H:
                    pad = H - h2
                    right = cv2.copyMakeBorder(right, 0, pad, 0, 0, cv2.BORDER_CONSTANT)
                side = np.hstack([left, right])
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(self.output_dir, f"pointmap_{ts}.png")
                cv2.imwrite(out_path, side)
                print(f"Saved snapshot: {os.path.relpath(out_path, self.project_root)}")

        cv2.destroyAllWindows()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Tool #7 - Point Mapper (Original → Rectified)")
    parser.add_argument("--fisheye-json", default="data/GPS-Real_fisheye_calibration.json", help="Path to fisheye calibration JSON")
    parser.add_argument("--transform-json", default="data/GPS-Real_corrected_transform.json", help="Path to rectification transform JSON")
    parser.add_argument("--server-width", type=int, default=2048, help="Server original image width (e.g., 2048)")
    parser.add_argument("--server-height", type=int, default=1536, help="Server original image height (e.g., 1536)")
    parser.add_argument("--scale-factor", type=float, default=0.8, help="Scale factor used in corrected camera matrix (default 0.8)")
    args = parser.parse_args()

    try:
        app = PointMapperApp(
            fisheye_json=args.fisheye_json,
            transform_json=args.transform_json,
            server_w=args.server_width,
            server_h=args.server_height,
            scale_factor=args.scale_factor,
        )
        return app.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
