#!/usr/bin/env python3
"""
Arena rectification tool - warps detected arena corners into an orientation-aligned
rectangle (not necessarily square) and estimates approximate orientation (yaw, pitch, roll).

Usage: python tools/rectify_arena_square.py <corner_json_path>

This tool:
1. Loads corner data JSON produced by the corner detector tool
2. Computes a homography to map the arena quadrilateral to an oriented rectangle
3. Warps the source image to that rectangle and saves it to output/
4. Estimates approximate yaw/pitch/roll from the homography (assumes intrinsics)
5. Saves transform metadata (homographies, angles) to data/

Notes:
- The corners may be outside the original uncorrected image due to fisheye expansion.
  This tool uses the corrected source image dimensions for all transforms.
- If --size is provided, a square of side N is used instead of a rectangle.
"""

import os
import json
import math
import argparse
from typing import Dict, Tuple

import cv2
import numpy as np


def read_corners_json(json_path: str) -> Dict:
    with open(json_path, 'r') as f:
        data = json.load(f)
    if 'detected_corners' not in data or not data['detected_corners']:
        raise ValueError("JSON does not contain 'detected_corners'.")
    return data


def order_corners(dc: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Return corners in TL, TR, BR, BL order as float32 array shape (4,2)."""
    required = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
    for k in required:
        if k not in dc:
            raise ValueError(f"Missing corner '{k}' in detected_corners")
    pts = np.array([
        dc['top_left'],
        dc['top_right'],
        dc['bottom_right'],
        dc['bottom_left'],
    ], dtype=np.float32)
    return pts


def compute_target_rectangle(src_corners: np.ndarray) -> Tuple[int, int]:
    """Compute a reasonable oriented rectangle size (width, height) from edge lengths."""
    tl, tr, br, bl = src_corners
    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    height_right = float(np.linalg.norm(br - tr))
    height_left = float(np.linalg.norm(bl - tl))
    avg_w = 0.5 * (width_top + width_bottom)
    avg_h = 0.5 * (height_left + height_right)
    # enforce sensible minimums
    W = int(round(max(256.0, avg_w)))
    H = int(round(max(256.0, avg_h)))
    return W, H


def decompose_homography_to_ypr(H_w2i: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[float, float, float]:
    """
    Estimate yaw(Z), pitch(Y), roll(X) in degrees from world->image homography.
    Assumptions:
    - Intrinsics K are approximated: fx=fy=max(w,h), cx=w/2, cy=h/2
    - World plane Z=0, H ~ K [r1 r2 t]
    Returns (yaw_deg, pitch_deg, roll_deg) using ZYX convention where
    R = Rz(yaw) * Ry(pitch) * Rx(roll).
    """
    h, w = image_shape
    fx = fy = float(max(w, h))
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    # Normalize H so that K^-1 H has columns r1, r2 of unit norm
    H = H_w2i.astype(np.float64)
    # Remove scale using first two columns
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]
    lam = 1.0 / np.linalg.norm(K_inv @ h1)
    r1 = lam * (K_inv @ h1)
    r2 = lam * (K_inv @ h2)
    r3 = np.cross(r1, r2)
    R_approx = np.column_stack((r1, r2, r3))
    # Orthonormalize
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt

    # ZYX Euler extraction
    # pitch = asin(-R[2,0]); roll = atan2(R[2,1], R[2,2]); yaw = atan2(R[1,0], R[0,0])
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])

    yaw_deg = math.degrees(yaw)
    pitch_deg = math.degrees(pitch)
    roll_deg = math.degrees(roll)
    return yaw_deg, pitch_deg, roll_deg


def warp_bounds_for_canvas(H_i2w: np.ndarray, src_w: int, src_h: int) -> Tuple[int, int, int, int]:
    """Compute axis-aligned bounds of the warped source image in output space.
    Returns (min_x, min_y, max_x, max_y)."""
    # Corners of the source image
    corners = np.array([
        [0, 0],
        [src_w - 1, 0],
        [src_w - 1, src_h - 1],
        [0, src_h - 1]
    ], dtype=np.float32)
    corners_h = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H_i2w).reshape(-1, 2)
    xs = corners_h[:, 0]
    ys = corners_h[:, 1]
    min_x = int(math.floor(xs.min()))
    max_x = int(math.ceil(xs.max()))
    min_y = int(math.floor(ys.min()))
    max_y = int(math.ceil(ys.max()))
    return min_x, min_y, max_x, max_y


def rectify_arena(corners_json_path: str, width_arg: int = 0, height_arg: int = 0, size_arg: int = 0, margin_arg: int = 0) -> int:
    data = read_corners_json(corners_json_path)

    # Resolve project root and ensure output directories exist
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    output_dir = os.path.join(project_root, 'output')
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Load source image (corrected), be resilient to relative paths
    src_image_path = data.get('source_image') or data.get('corrected_image')
    if not src_image_path:
        raise ValueError("Corner JSON must include 'source_image' or 'corrected_image'.")
    if not os.path.isabs(src_image_path):
        candidate = os.path.join(project_root, src_image_path)
        if os.path.exists(candidate):
            src_image_path = candidate

    image = cv2.imread(src_image_path)
    if image is None:
        raise ValueError(f"Could not load source image: {src_image_path}")
    h, w = image.shape[:2]

    # Read and order corners
    src_corners = order_corners(data['detected_corners'])

    # Decide target output geometry
    if size_arg and size_arg > 0:
        # Square output if --size is used
        W = H = int(size_arg)
    else:
        # Oriented rectangle by default (can be overridden by --width/--height)
        default_W, default_H = compute_target_rectangle(src_corners)
        W = int(width_arg) if width_arg and width_arg > 0 else default_W
        H = int(height_arg) if height_arg and height_arg > 0 else default_H
    dst_rect = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)

    # Homography world->image: map oriented rectangle to the image quad
    world_rect = dst_rect.copy()
    H_w2i = cv2.getPerspectiveTransform(world_rect, src_corners)
    # Inverse for warping image->square
    H_i2w = np.linalg.inv(H_w2i)

    # Compute bounds of the entire warped source so we don't crop content
    min_x, min_y, max_x, max_y = warp_bounds_for_canvas(H_i2w, w, h)
    # Optional extra margin around
    extra = int(max(0, margin_arg))
    min_x -= extra
    min_y -= extra
    max_x += extra
    max_y += extra
    # Translate so that min_x/min_y maps to (0,0)
    T = np.array([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]], dtype=np.float64)
    H_i2w_canvas = T @ H_i2w
    canvas_W = int(max(W, max_x - min_x))
    canvas_H = int(max(H, max_y - min_y))
    # Warp to canvas big enough to contain the whole warped image
    rectified = cv2.warpPerspective(image, H_i2w_canvas, (canvas_W, canvas_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Overlay dashed red border (30% transparency) showing the oriented arena rectangle
    def draw_dashed_line(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 2, dash_length: int = 12, alpha: float = 0.3) -> None:
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        dx = x2 - x1
        dy = y2 - y1
        length = float(math.hypot(dx, dy))
        if length == 0:
            return
        ux = dx / length
        uy = dy / length
        overlay = img.copy()
        current = 0.0
        draw_segment = True
        while current < length:
            if draw_segment:
                sx = int(round(x1 + ux * current))
                sy = int(round(y1 + uy * current))
                end = min(current + dash_length, length)
                ex = int(round(x1 + ux * end))
                ey = int(round(y1 + uy * end))
                cv2.line(overlay, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
            current += dash_length
            draw_segment = not draw_segment
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Rectangle corners in canvas coordinates after translation T
    rect_tl = (int(round(0 - min_x)), int(round(0 - min_y)))
    rect_tr = (int(round((W - 1) - min_x)), int(round(0 - min_y)))
    rect_br = (int(round((W - 1) - min_x)), int(round((H - 1) - min_y)))
    rect_bl = (int(round(0 - min_x)), int(round((H - 1) - min_y)))

    border_color = (0, 0, 255)  # Red (BGR)
    draw_dashed_line(rectified, rect_tl, rect_tr, border_color, thickness=2, dash_length=12, alpha=0.3)
    draw_dashed_line(rectified, rect_tr, rect_br, border_color, thickness=2, dash_length=12, alpha=0.3)
    draw_dashed_line(rectified, rect_br, rect_bl, border_color, thickness=2, dash_length=12, alpha=0.3)
    draw_dashed_line(rectified, rect_bl, rect_tl, border_color, thickness=2, dash_length=12, alpha=0.3)

    # Estimate approximate yaw/pitch/roll from H_w2i
    yaw_deg, pitch_deg, roll_deg = decompose_homography_to_ypr(H_w2i, (h, w))

    # Build output filenames
    base = os.path.splitext(os.path.basename(src_image_path))[0]
    rectified_filename = f"{base}_rectified_oriented.png"
    rectified_path = os.path.join(output_dir, rectified_filename)
    transform_filename = f"{base}_transform.json"
    transform_path = os.path.join(data_dir, transform_filename)

    # Save image and transform
    cv2.imwrite(rectified_path, rectified)
    rectified_rel = os.path.relpath(rectified_path, project_root)
    transform_rel = os.path.relpath(transform_path, project_root)

    transform_data = {
        "source_image": os.path.relpath(src_image_path, project_root) if os.path.isabs(src_image_path) else src_image_path,
        "rectified_image": rectified_rel,
        "input_corners": {
            "top_left": src_corners[0].tolist(),
            "top_right": src_corners[1].tolist(),
            "bottom_right": src_corners[2].tolist(),
            "bottom_left": src_corners[3].tolist(),
        },
        "output_size": {"width": W, "height": H},
        "homography_world_to_image": H_w2i.tolist(),
        "homography_image_to_world": H_i2w.tolist(),
        "homography_image_to_world_canvas": H_i2w_canvas.tolist(),
        "warp_bounds": {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y
        },
        "intrinsics_assumed": {
            "fx": float(max(w, h)),
            "fy": float(max(w, h)),
            "cx": w / 2.0,
            "cy": h / 2.0,
        },
        "orientation_estimate_degrees": {
            "yaw_z": yaw_deg,
            "pitch_y": pitch_deg,
            "roll_x": roll_deg,
        },
        "notes": "Yaw(Z), Pitch(Y), Roll(X) estimated via homography decomposition with assumed intrinsics.",
    }

    with open(transform_path, 'w') as f:
        json.dump(transform_data, f, indent=2)

    # Summary
    print(f"Saved rectified image: {rectified_rel}")
    print(f"Saved transform JSON: {transform_rel}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Rectify arena to a square and estimate orientation")
    parser.add_argument("corner_json", help="Path to the corner JSON from the corner detector tool")
    parser.add_argument("--width", type=int, default=0, help="Desired output width (oriented rectangle)")
    parser.add_argument("--height", type=int, default=0, help="Desired output height (oriented rectangle)")
    parser.add_argument("--size", type=int, default=0, help="Force square side length (overrides width/height)")
    parser.add_argument("--margin", type=int, default=0, help="Extra margin (pixels) around warped canvas to avoid cropping")
    args = parser.parse_args()

    try:
        return rectify_arena(args.corner_json, args.width, args.height, args.size, args.margin)
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
