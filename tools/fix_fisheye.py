#!/usr/bin/env python3
"""
Fisheye correction tool - focuses only on detecting and correcting fisheye distortion.

Usage: python fix_fisheye.py <image_path>

This tool:
1. Lets you click points on curved wall edges in the fisheye image
2. Uses OpenCV's fisheye calibration to detect distortion parameters
3. Applies correction with expanded margins to show more of the arena
4. Saves the corrected image and calibration data
"""

import cv2
import numpy as np
import json
import os
import argparse
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict

# Optional Qt viewer for hidden-cursor mode
try:
    from src.tools.qt_viewer import run_viewer as qt_run_viewer
    _HAVE_QT = True
except Exception:
    _HAVE_QT = False


@dataclass
class FisheyeCalibration:
    """OpenCV fisheye calibration parameters"""
    camera_matrix: List[List[float]]  # 3x3 camera intrinsic matrix
    distortion_coeffs: List[float]    # 4 fisheye distortion coefficients [k1, k2, k3, k4]
    image_size: Tuple[int, int]       # Original image size (width, height)
    corrected_size: Tuple[int, int]   # Corrected image size (width, height)
    margin_pixels: int                # Margin added around image


@dataclass
class WallPoints:
    """Points clicked along a wall segment"""
    wall_name: str
    points: List[Tuple[int, int]]
    color: Tuple[int, int, int]


class FisheyeCorrector:
    """Tool for fisheye distortion correction"""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        self.display_image = self.original_image.copy()
        self.height, self.width = self.original_image.shape[:2]
        
        # Viewport state for zoom/pan
        # view_zoom: magnification level (1.0 = fit image, >1.0 = zoom in)
        # view_center: center of the viewport in image space (x, y)
        self.view_zoom: float = 1.0
        self.view_center: Tuple[int, int] = (self.width // 2, self.height // 2)
        self.is_panning: bool = False
        self.last_mouse_pos: Tuple[int, int] = (0, 0)
        self.pan_button: Optional[str] = None
        self.cursor_pos: Tuple[int, int] = (self.width // 2, self.height // 2)
        # Track cursor in base-image coordinates for drawing a stable crosshair
        self.cursor_base_xy: Tuple[int, int] = (self.width // 2, self.height // 2)
        # Cached ROI from last render for fast screen->image mapping
        # (x0, y0, roi_w, roi_h, view_w, view_h)
        self.current_view_roi: Optional[Tuple[int, int, int, int, int, int]] = None
        
        # Wall segments for point collection
        self.walls = [
            WallPoints("left", [], (0, 255, 0)),    # Green
            WallPoints("right", [], (255, 0, 0)),   # Blue  
            WallPoints("top", [], (0, 0, 255)),     # Red
            WallPoints("bottom", [], (255, 255, 0)) # Cyan
        ]
        self.current_wall_idx = 0
        
        # Correction state
        self.fisheye_calibration: Optional[FisheyeCalibration] = None
        self.corrected_image = None
        self.show_corrected = False  # Start with original during point collection
        self.auto_corrected = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for point collection and navigation (zoom/pan)"""
        cursor_events = (
            cv2.EVENT_MOUSEMOVE,
            cv2.EVENT_LBUTTONDOWN,
            cv2.EVENT_RBUTTONDOWN,
            cv2.EVENT_MBUTTONDOWN,
        )
        wheel_event = getattr(cv2, 'EVENT_MOUSEWHEEL', None)
        if wheel_event is not None:
            cursor_events += (wheel_event,)
        if event in cursor_events:
            self.cursor_pos = (x, y)
            bx, by = self._screen_to_base_xy(x, y)
            if bx is not None and by is not None:
                self.cursor_base_xy = (bx, by)
            # Refresh view to move crosshair smoothly with the mouse
            if event == cv2.EVENT_MOUSEMOVE:
                self.update_display()

        dbl_left = getattr(cv2, 'EVENT_LBUTTONDBLCLK', None)
        dbl_right = getattr(cv2, 'EVENT_RBUTTONDBLCLK', None)
        if dbl_left is not None and event == dbl_left:
            self._zoom_at_screen_point(1, x, y)
            self.update_display()
            return
        if dbl_right is not None and event == dbl_right:
            self._zoom_at_screen_point(-1, x, y)
            self.update_display()
            return
        if self._handle_pan(event, x, y, flags):
            return
        if self._handle_zoom(event, x, y, flags):
            return
        if self._handle_add_point(event, x, y):
            return

    def _handle_pan(self, event, x, y, flags=0):
        """Handle middle mouse drag to pan."""
        shift_flag = getattr(cv2, 'EVENT_FLAG_SHIFTKEY', 0)
        ctrl_flag = getattr(cv2, 'EVENT_FLAG_CTRLKEY', 0)
        alt_flag = getattr(cv2, 'EVENT_FLAG_ALTKEY', 0)
        modifier_active = False
        for flag in (shift_flag, ctrl_flag, alt_flag):
            if flag and (flags & flag):
                modifier_active = True
                break

        if event == cv2.EVENT_MBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and modifier_active):
            self.is_panning = True
            if event == cv2.EVENT_MBUTTONDOWN:
                self.pan_button = 'middle'
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.pan_button = 'right'
            else:
                self.pan_button = 'left'
            self.last_mouse_pos = (x, y)
            return True
        elif event in (cv2.EVENT_MBUTTONUP, cv2.EVENT_RBUTTONUP) or (event == cv2.EVENT_LBUTTONUP and self.pan_button == 'left'):
            self.is_panning = False
            self.pan_button = None
            return True
        elif event == cv2.EVENT_MOUSEMOVE and self.is_panning:
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]
            self.last_mouse_pos = (x, y)
            self._pan_by_pixels(dx, dy)
            self.update_display()
            return True
        return False

    def _handle_zoom(self, event, x, y, flags):
        """Handle mouse wheel to zoom (centered at cursor)."""
        if event == getattr(cv2, 'EVENT_MOUSEWHEEL', -1):
            direction = 1 if flags > 0 else -1
            self._zoom_at_screen_point(direction, x, y)
            self.update_display()
            return True
        return False

    def _handle_add_point(self, event, x, y):
        """Handle left click to add a point (only in original view)."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.show_corrected:
                print("Switch to ORIGINAL view ('t') to add points")
                return True
            if self.current_wall_idx < len(self.walls):
                bx, by = self._screen_to_base_xy(x, y)
                if bx is None:
                    return True
                px, py = self._base_to_original_xy(bx, by)
                if px is None:
                    return True
                px = int(max(0, min(self.width - 1, px)))
                py = int(max(0, min(self.height - 1, py)))
                self.walls[self.current_wall_idx].points.append((px, py))
                wall_name = self.walls[self.current_wall_idx].wall_name
                num_points = len(self.walls[self.current_wall_idx].points)
                print(f"Added point ({px}, {py}) to {wall_name} wall [{num_points} points]")
                self.update_display()
                return True
        return False
                
    def undo_last_point(self):
        """Undo the last point added to current wall"""
        if self.current_wall_idx < len(self.walls):
            current_wall = self.walls[self.current_wall_idx]
            if len(current_wall.points) > 0:
                removed_point = current_wall.points.pop()
                wall_name = current_wall.wall_name
                remaining_points = len(current_wall.points)
                print(f"Undid point {removed_point} from {wall_name} wall [{remaining_points} points remaining]")
                self.update_display()
            else:
                print(f"No points to undo on {self.walls[self.current_wall_idx].wall_name} wall")
                
    def advance_to_next_wall(self):
        """Advance to next wall"""
        if self.current_wall_idx >= len(self.walls):
            return

        current_wall = self.walls[self.current_wall_idx]
        wall_name = current_wall.wall_name
        points_count = len(current_wall.points)

        if points_count >= 2:
            self._handle_wall_complete(wall_name, points_count)
        elif points_count >= 1:
            self._handle_wall_partial(wall_name, points_count)
        else:
            self._handle_wall_empty(wall_name)

    def _move_to_next_wall(self, message: Optional[str] = None):
        if self.current_wall_idx < len(self.walls) - 1:
            self.current_wall_idx += 1
            next_wall = self.walls[self.current_wall_idx].wall_name
            if next_wall == "top":
                print(f"Now click points on the {next_wall} wall - THIS IS THE MOST IMPORTANT!")
                print("Click many points along the curved top edge for best results")
            else:
                print(f"Now click points on the {next_wall} wall (even small visible portions help)")
                print("Press 'n' when done, even with just a few points")
            if message:
                print(message)
        else:
            print("✓ All walls complete! Calculating fisheye correction...")
            self.auto_calculate_correction()

    def _handle_wall_complete(self, wall_name, points_count):
        print(f"✓ {wall_name} wall complete! ({points_count} points collected)")
        self._move_to_next_wall()

    def _handle_wall_partial(self, wall_name, points_count):
        print(f"Only {points_count} point(s) for {wall_name} wall - that's okay for fisheye!")
        print("Even partial wall data helps. Moving to next wall...")
        self._move_to_next_wall()

    def _handle_wall_empty(self, wall_name):
        print(f"No points for {wall_name} wall - that's okay, moving to next wall")
        self._move_to_next_wall()
                    
    def update_display(self):
        """Update the display with current points and apply zoom/pan viewport"""
        base = self._prepare_base_image()
        view = self._apply_viewport(base)
        # Draw a subtle crosshair at the cursor for consistent pointer visuals
        self._draw_crosshair(view)
        self.display_image = view

    def _prepare_base_image(self) -> np.ndarray:
        """Return a copy of the image base with any overlays applied."""
        if self.show_corrected and self.corrected_image is not None:
            return self.corrected_image.copy()

        if hasattr(self, 'original_with_margins'):
            base = self.original_with_margins.copy()
            margin = self.fisheye_calibration.margin_pixels if self.fisheye_calibration else 0
        else:
            base = self.original_image.copy()
            margin = 0

        self._draw_wall_points(base, margin)
        return base

    def _draw_wall_points(self, image: np.ndarray, margin: int) -> None:
        """Draw collected wall points and connecting polylines."""
        for wall in self.walls:
            if not wall.points:
                continue

            offset_points = [(point[0] + margin, point[1] + margin) for point in wall.points]

            for point in offset_points:
                cv2.circle(image, point, 3, wall.color, -1)

            if len(offset_points) > 1:
                pts = np.array(offset_points, np.int32)
                cv2.polylines(image, [pts], False, wall.color, 3)

    # -------------------- Viewport helpers --------------------
    def _apply_viewport(self, image: np.ndarray) -> np.ndarray:
        """Apply zoom/pan to the given image and cache ROI for mapping."""
        h, w = image.shape[:2]
        # Ensure center is valid for current image size
        self._clamp_view_center(w, h)
        # Compute ROI size at current zoom
        zoom = max(1.0, float(self.view_zoom))
        roi_w = max(1, int(round(w / zoom)))
        roi_h = max(1, int(round(h / zoom)))
        half_w = roi_w // 2
        half_h = roi_h // 2
        cx, cy = self.view_center
        x0 = int(max(0, min(w - roi_w, cx - half_w)))
        y0 = int(max(0, min(h - roi_h, cy - half_h)))
        roi = image[y0:y0 + roi_h, x0:x0 + roi_w]
        # Upscale ROI back to window size so window dims remain constant for this image
        view = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)
        self.current_view_roi = (x0, y0, roi_w, roi_h, w, h)
        return view

    def _draw_crosshair(self, view_image: np.ndarray) -> None:
        """Draw a small crosshair at the current cursor position mapped into the view image."""
        if self.current_view_roi is None or view_image is None:
            return
        x0, y0, roi_w, roi_h, view_w, view_h = self.current_view_roi
        bx, by = self.cursor_base_xy
        # Map base → view coordinates
        vx = int(round((bx - x0) * float(view_w) / max(1.0, float(roi_w))))
        vy = int(round((by - y0) * float(view_h) / max(1.0, float(roi_h))))
        # Draw crosshair (clamped inside view)
        h, w = view_image.shape[:2]
        vx = max(0, min(w - 1, vx))
        vy = max(0, min(h - 1, vy))
        size = 10
        color = (255, 255, 255)
        thickness = 1
        cv2.line(view_image, (max(0, vx - size), vy), (min(w - 1, vx + size), vy), color, thickness, cv2.LINE_AA)
        cv2.line(view_image, (vx, max(0, vy - size)), (vx, min(h - 1, vy + size)), color, thickness, cv2.LINE_AA)

    def _clamp_view_center(self, w: int, h: int) -> None:
        """Clamp view center so the ROI stays inside image bounds."""
        zoom = max(1.0, float(self.view_zoom))
        roi_w = max(1, int(round(w / zoom)))
        roi_h = max(1, int(round(h / zoom)))
        min_cx = roi_w // 2
        max_cx = w - (roi_w - roi_w // 2)
        min_cy = roi_h // 2
        max_cy = h - (roi_h - roi_h // 2)
        cx, cy = self.view_center
        cx = int(max(min_cx, min(max_cx, cx)))
        cy = int(max(min_cy, min(max_cy, cy)))
        self.view_center = (cx, cy)

    def _screen_to_base_xy(self, sx: int, sy: int) -> Tuple[Optional[int], Optional[int]]:
        """Map screen (window) coordinates to base image pixel coordinates using current ROI."""
        if self.current_view_roi is None:
            return None, None
        x0, y0, roi_w, roi_h, view_w, view_h = self.current_view_roi
        # Map proportionally from view pixel to ROI
        bx = x0 + (sx * roi_w) / max(1, view_w)
        by = y0 + (sy * roi_h) / max(1, view_h)
        return int(round(bx)), int(round(by))

    def _base_to_original_xy(self, bx: int, by: int) -> Tuple[Optional[int], Optional[int]]:
        """Convert base image coords to original image coords (remove margins if present)."""
        if hasattr(self, 'original_with_margins') and not self.show_corrected and self.fisheye_calibration is not None:
            margin = self.fisheye_calibration.margin_pixels
            return bx - margin, by - margin
        # Already in original coords
        return bx, by

    def _zoom_at_screen_point(self, direction: int, sx: int, sy: int) -> None:
        """Zoom in/out with focus kept at the cursor position."""
        if self.display_image is None:
            return
        # Convert cursor position to base coordinates before zoom
        pre_bx, pre_by = self._screen_to_base_xy(sx, sy)
        # Adjust zoom
        factor = 1.25 if direction > 0 else (1.0 / 1.25)
        new_zoom = self.view_zoom * factor
        # Clamp zoom between 1x and 10x
        self.view_zoom = float(max(1.0, min(10.0, new_zoom)))
        # Convert cursor position after zoom and shift center to keep point stable
        post_bx, post_by = self._screen_to_base_xy(sx, sy)
        if pre_bx is not None and post_bx is not None:
            dx = pre_bx - post_bx
            dy = pre_by - post_by
            cx, cy = self.view_center
            self.view_center = (int(round(cx + dx)), int(round(cy + dy)))

    def _pan_by_pixels(self, dx_view: int, dy_view: int) -> None:
        """Pan the viewport by a delta in screen pixels, converted to image pixels."""
        if self.current_view_roi is None:
            return
        _, _, roi_w, roi_h, view_w, view_h = self.current_view_roi
        # Convert screen movement to base-image movement
        move_x = (dx_view * roi_w) / max(1, view_w)
        move_y = (dy_view * roi_h) / max(1, view_h)
        cx, cy = self.view_center
        # Dragging the image should move content with the cursor (invert movement)
        self.view_center = (int(round(cx - move_x)), int(round(cy - move_y)))
        # Clamp will occur in next render

    def _pan_by_image_units(self, dx_image: int, dy_image: int) -> None:
        """Pan the viewport by a delta expressed directly in image pixels."""
        if dx_image == 0 and dy_image == 0:
            return
        cx, cy = self.view_center
        self.view_center = (int(round(cx + dx_image)), int(round(cy + dy_image)))
        self.update_display()

    @staticmethod
    def _key_in(key: int, raw_key: int, *chars: str) -> bool:
        """Return True if either key code matches any provided characters."""
        for ch in chars:
            code = ord(ch)
            if key == code:
                return True
            if raw_key not in (-1, None) and (raw_key & 0xFF) == code:
                return True
        return False

    def _handle_keyboard_pan(self, key: int, raw_key: int) -> bool:
        """Handle keyboard panning; returns True if the key was consumed."""
        mapping = {
            'a': "left",
            'd': "right",
            'w': "up",
            's': "down",
        }

        direction = None
        if key not in (-1, None):
            try:
                char = chr(key).lower()
            except ValueError:
                char = ''
            direction = mapping.get(char)

        if direction is None and raw_key not in (-1, None):
            try:
                char = chr(raw_key & 0xFF).lower()
            except ValueError:
                char = ''
            direction = mapping.get(char)
        if direction is None:
            return False

        if self.current_view_roi is not None:
            _, _, roi_w, roi_h, _, _ = self.current_view_roi
        else:
            roi_w, roi_h = self.width, self.height

        zoom = max(1.0, float(self.view_zoom))
        step_x = max(5, int(round((roi_w * 0.08) / zoom)))
        step_y = max(5, int(round((roi_h * 0.08) / zoom)))

        dx_image = dy_image = 0
        if direction == "left":
            dx_image = -step_x
        elif direction == "right":
            dx_image = step_x
        elif direction == "up":
            dy_image = -step_y
        elif direction == "down":
            dy_image = step_y

        if dx_image == 0 and dy_image == 0:
            return False

        self._pan_by_image_units(dx_image, dy_image)
        return True
                        
    def calibrate_fisheye_opencv(self, margin_pixels: int = 200) -> Optional[FisheyeCalibration]:
        """Use OpenCV's fisheye undistortion with estimated parameters"""
        print("Using OpenCV fisheye undistortion...")
        
        # Collect all wall points
        all_points = []
        for wall in self.walls:
            if len(wall.points) >= 2:
                all_points.extend(wall.points)
                
        if len(all_points) < 4:
            print("Need at least 4 points total for fisheye calibration")
            return None
            
        print(f"Using {len(all_points)} points for calibration")
        
        # Check if we have good top wall data
        top_wall_points = len([p for wall in self.walls if wall.wall_name == "top" for p in wall.points])
        side_wall_points = len(all_points) - top_wall_points
        
        print(f"  Top wall: {top_wall_points} points (most important for fisheye)")
        print(f"  Side walls: {side_wall_points} points")
        
        if top_wall_points >= 8:
            print("✓ Good top wall data - this will give excellent fisheye correction!")
        elif top_wall_points >= 4:
            print("✓ Decent top wall data - should give good fisheye correction")
        else:
            print("⚠ Limited top wall data - correction may be approximate")
            
        # For fisheye images, estimate camera parameters
        focal_length = min(self.width, self.height) * 0.8  # Typical for fisheye
        
        camera_matrix = np.array([
            [focal_length, 0, self.width/2],
            [0, focal_length, self.height/2], 
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Estimate fisheye distortion coefficients based on wall curvature
        # Focus on the top wall since it's most visible in fisheye images
        total_curvature = 0.0
        wall_count = 0
        
        for wall in self.walls:
            if len(wall.points) >= 3:
                # Measure curvature of this wall
                points = np.array(wall.points, dtype=np.float32)
                
                # Fit a line and measure deviation
                vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
                
                # Calculate average distance from line
                total_deviation = 0.0
                for px, py in wall.points:
                    # Distance from point to line
                    distance = abs(vx[0] * (py - y0[0]) - vy[0] * (px - x0[0]))
                    total_deviation += distance
                    
                avg_deviation = total_deviation / len(wall.points)
                # Convert to relative curvature
                image_diagonal = np.sqrt(self.width**2 + self.height**2)
                relative_curvature = avg_deviation / image_diagonal
                
                # Weight the top wall more heavily since it's most visible
                if wall.wall_name == "top":
                    weight = 3.0  # Top wall is 3x more important
                    print(f"Top wall curvature: {relative_curvature:.4f} (weighted 3x)")
                else:
                    weight = 1.0
                    print(f"{wall.wall_name} wall curvature: {relative_curvature:.4f}")
                
                total_curvature += relative_curvature * weight
                wall_count += weight
        
        if wall_count > 0:
            avg_curvature = total_curvature / wall_count
            # For fisheye, use a stronger correction factor
            k1 = avg_curvature * 8.0  # Increased scale factor for fisheye
            k1 = min(k1, 1.5)  # Allow stronger correction for fisheye
            k1 = max(k1, 0.1)  # Minimum correction
        else:
            k1 = 0.5  # Default stronger fisheye distortion
            
        # Fisheye distortion coefficients [k1, k2, k3, k4]
        dist_coeffs = np.array([k1, k1*0.3, 0.0, 0.0], dtype=np.float32)
        
        print(f"Estimated fisheye parameters:")
        print(f"  Camera: fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
        print(f"  Center: cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
        print(f"  Distortion: k1={dist_coeffs[0]:.4f}, k2={dist_coeffs[1]:.4f}")
        
        # Calculate expanded output size
        expanded_size = (self.width + 2 * margin_pixels, self.height + 2 * margin_pixels)
        print(f"Expanding output from ({self.width}, {self.height}) to {expanded_size} with {margin_pixels}px margins")
        
        return FisheyeCalibration(
            camera_matrix=camera_matrix.tolist(),
            distortion_coeffs=dist_coeffs.tolist(),
            image_size=(self.width, self.height),
            corrected_size=expanded_size,
            margin_pixels=margin_pixels
        )
        
    def apply_fisheye_correction(self, image: np.ndarray, fisheye_cal: FisheyeCalibration) -> np.ndarray:
        """Apply fisheye correction using OpenCV's undistortion with expanded view"""
        camera_matrix = np.array(fisheye_cal.camera_matrix, dtype=np.float32)
        dist_coeffs = np.array(fisheye_cal.distortion_coeffs, dtype=np.float32)
        
        # Create expanded output size
        expanded_size = fisheye_cal.corrected_size
        margin = fisheye_cal.margin_pixels
        
        # Adjust camera matrix for the expanded image (shift principal point)
        new_camera_matrix = camera_matrix.copy()
        new_camera_matrix[0, 2] += margin  # cx offset
        new_camera_matrix[1, 2] += margin  # cy offset
        
        # Reduce focal length slightly to show more area
        scale_factor = 0.8  # Show more area
        new_camera_matrix[0, 0] *= scale_factor  # fx
        new_camera_matrix[1, 1] *= scale_factor  # fy
        
        # Create undistortion maps for expanded size
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix,
            expanded_size, cv2.CV_16SC2
        )
        
        # Apply undistortion using remap to expanded size
        undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        
        return undistorted
        
    def add_margins_to_image(self, image: np.ndarray, margin: int, target_size: Tuple[int, int]) -> np.ndarray:
        """Add margins around image to match target size"""
        target_width, target_height = target_size
        
        # Create a black canvas of target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate position to center the original image
        start_x = margin
        start_y = margin
        end_x = start_x + image.shape[1]
        end_y = start_y + image.shape[0]
        
        # Place original image in center of canvas
        canvas[start_y:end_y, start_x:end_x] = image
        
        return canvas
        
    def auto_calculate_correction(self):
        """Automatically calculate distortion correction when all walls are complete"""
        if not self.auto_corrected:
            print("\n" + "="*60)
            print("CALCULATING FISHEYE CORRECTION")
            print("="*60)
            print("Using all wall points to detect fisheye distortion...")
            
            # Count total points
            total_points = sum(len(wall.points) for wall in self.walls)
            print(f"Total points collected: {total_points}")
            print("\nThis may take a moment - please wait...")
            print("Using OpenCV's professional fisheye calibration!")
            
            self.fisheye_calibration = self.calibrate_fisheye_opencv()
            
            if self.fisheye_calibration is None:
                print("Fisheye calibration failed")
                return
                
            print("\nApplying fisheye correction to image...")
            self.corrected_image = self.apply_fisheye_correction(self.original_image, self.fisheye_calibration)
            
            # Create original image with same margins for seamless toggle
            margin = self.fisheye_calibration.margin_pixels
            expanded_size = self.fisheye_calibration.corrected_size
            self.original_with_margins = self.add_margins_to_image(self.original_image, margin, expanded_size)
            
            self.show_corrected = True  # Switch to corrected view after completion
            self.auto_corrected = True
            # Reset viewport for new image size
            self.view_zoom = 1.0
            self.view_center = (expanded_size[0] // 2, expanded_size[1] // 2)
            self.update_display()  # Update display to show corrected view
            
            # Resize window to fit corrected image
            corrected_width, corrected_height = self.fisheye_calibration.corrected_size
            cv2.resizeWindow("Fisheye Correction", corrected_width, corrected_height)
            
            print(f"✓ OpenCV Fisheye correction applied!")
            camera_matrix = self.fisheye_calibration.camera_matrix
            dist_coeffs = self.fisheye_calibration.distortion_coeffs
            print(f"  Camera: fx={camera_matrix[0][0]:.1f}, fy={camera_matrix[1][1]:.1f}")
            print(f"  Center: cx={camera_matrix[0][2]:.1f}, cy={camera_matrix[1][2]:.1f}")
            print(f"  Distortion: k1={dist_coeffs[0]:.4f}, k2={dist_coeffs[1]:.4f}, k3={dist_coeffs[2]:.4f}, k4={dist_coeffs[3]:.4f}")
            
            print("\n" + "="*60)
            print("✓ FISHEYE CORRECTION COMPLETE!")
            print("="*60)
            print("Now showing: CORRECTED VIEW (fisheye removed)")
            print("✓ Curved walls are now straight lines")
            print("✓ Arena corners should be visible")
            print("✓ Both images now have same size for seamless toggle")
            print()
            print("Controls:")
            print("  't' = Toggle between corrected ↔ original views")
            print("  's' = Save corrected image and calibration data")
            print("  'q' = Quit")
            
    def save_calibration_and_image(self):
        """Save the corrected image and calibration data"""
        if not self.fisheye_calibration or self.corrected_image is None:
            print("No calibration data to save")
            return
            
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        
        # Resolve project root and ensure output directories exist
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        output_dir = os.path.join(project_root, 'output')
        data_dir = os.path.join(project_root, 'data')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Save corrected image into output/
        corrected_filename = f"{base_name}_corrected.png"
        corrected_path = os.path.join(output_dir, corrected_filename)
        corrected_rel = os.path.relpath(corrected_path, project_root)
        image_saved = cv2.imwrite(corrected_path, self.corrected_image)
        if image_saved:
            print(f"✓ Saved corrected image: {corrected_rel}")
        else:
            print(f"✗ Failed to save corrected image: {corrected_rel}")
            print("Please check write permissions and available disk space before retrying.")
            return
        
        # Save calibration data into data/
        calibration_data = {
            "original_image": self.image_path,
            "corrected_image": corrected_rel,
            "fisheye_calibration": asdict(self.fisheye_calibration),
            "wall_points": [
                {
                    "wall_name": wall.wall_name,
                    "points": wall.points,
                    "num_points": len(wall.points)
                }
                for wall in self.walls if wall.points
            ]
        }
        
        calibration_filename = f"{base_name}_fisheye_calibration.json"
        calibration_path = os.path.join(data_dir, calibration_filename)
        calibration_rel = os.path.relpath(calibration_path, project_root)
        try:
            with open(calibration_path, 'w') as f:
                json.dump(calibration_data, f, indent=2)
        except OSError as exc:
            print(f"✗ Failed to save calibration data: {calibration_rel} ({exc})")
            print("Corrected image was saved successfully; retry the save once the filesystem issue is resolved.")
            return
        print(f"✓ Saved calibration data: {calibration_rel}")
        
        # Display summary
        print(f"\n" + "="*50)
        print("CALIBRATION SUMMARY")
        print("="*50)
        print(f"Original image: {self.width}x{self.height}")
        print(f"Corrected image: {self.fisheye_calibration.corrected_size[0]}x{self.fisheye_calibration.corrected_size[1]}")
        print(f"Margin added: {self.fisheye_calibration.margin_pixels}px")
        print(f"Total wall points: {sum(len(wall.points) for wall in self.walls)}")
        
        camera_matrix = self.fisheye_calibration.camera_matrix
        dist_coeffs = self.fisheye_calibration.distortion_coeffs
        print(f"\nCamera parameters:")
        print(f"  fx={camera_matrix[0][0]:.1f}, fy={camera_matrix[1][1]:.1f}")
        print(f"  cx={camera_matrix[0][2]:.1f}, cy={camera_matrix[1][2]:.1f}")
        print(f"Distortion coefficients:")
        print(f"  k1={dist_coeffs[0]:.4f}, k2={dist_coeffs[1]:.4f}")
        print(f"  k3={dist_coeffs[2]:.4f}, k4={dist_coeffs[3]:.4f}")
        
    def run(self):
        """Run the fisheye correction tool"""
        # Prefer Qt viewer (hidden OS cursor) if available
        if _HAVE_QT:
            print("Fisheye Correction Tool (Qt)")
            print("=" * 40)
            print("Instructions:")
            print("1. Click points along ANY visible wall edges (even just a few points help!)")
            print("2. Focus on the TOP wall - it's most visible and important")
            print("3. For side walls: click whatever small portions you can see")
            print("4. Mouse wheel zooms; Right/Middle drag pans; keys: n,z,t,r,s,q")
            print()

            pressed_btn = {"btn": None}

            def frame_provider() -> np.ndarray:
                self.update_display()
                return self.display_image

            def on_mouse(kind: str, mx: int, my: int, button_or_buttons: int, _mods: int, delta: int) -> None:
                if kind == "move":
                    self.mouse_callback(cv2.EVENT_MOUSEMOVE, mx, my, 0, None)
                elif kind == "press":
                    pressed_btn["btn"] = button_or_buttons
                    if button_or_buttons == 1:
                        self.mouse_callback(cv2.EVENT_LBUTTONDOWN, mx, my, 0, None)
                    elif button_or_buttons == 2:
                        self.mouse_callback(cv2.EVENT_RBUTTONDOWN, mx, my, 0, None)
                    elif button_or_buttons == 4:
                        self.mouse_callback(cv2.EVENT_MBUTTONDOWN, mx, my, 0, None)
                elif kind == "release":
                    btn = pressed_btn["btn"]
                    pressed_btn["btn"] = None
                    if btn == 1:
                        self.mouse_callback(cv2.EVENT_LBUTTONUP, mx, my, 0, None)
                    elif btn == 2:
                        self.mouse_callback(cv2.EVENT_RBUTTONUP, mx, my, 0, None)
                    elif btn == 4:
                        self.mouse_callback(cv2.EVENT_MBUTTONUP, mx, my, 0, None)
                elif kind == "wheel":
                    direction = 1 if delta > 0 else -1
                    self._zoom_at_screen_point(direction, mx, my)
                    self.update_display()

            def on_key(key: int) -> None:
                try:
                    ch = chr(key)
                except Exception:
                    ch = ''
                if ch.lower() == 'q':
                    from PyQt5 import QtWidgets
                    QtWidgets.QApplication.instance().quit()
                elif ch.lower() == 'z':
                    self.undo_last_point()
                elif ch in ['+', '=']:
                    cx, cy = self.cursor_pos
                    self._zoom_at_screen_point(1, cx, cy)
                    self.update_display()
                elif ch in ['-', '_']:
                    cx, cy = self.cursor_pos
                    self._zoom_at_screen_point(-1, cx, cy)
                    self.update_display()
                elif ch.lower() == 'n':
                    self.advance_to_next_wall()
                elif ch.lower() == 'r':
                    if self.current_wall_idx < len(self.walls):
                        wall_name = self.walls[self.current_wall_idx].wall_name
                        self.walls[self.current_wall_idx].points.clear()
                        self.update_display()
                        print(f"Reset {wall_name} wall points")
                elif ch.lower() == 'c':
                    if not self.auto_corrected:
                        print("Calculating distortion correction...")
                        self.auto_calculate_correction()
                elif ch.lower() == 't':
                    if self.corrected_image is not None:
                        self.show_corrected = not self.show_corrected
                        self.update_display()
                    else:
                        print("Complete fisheye correction first")
                elif ch.lower() == 's':
                    if self.corrected_image is not None:
                        self.save_calibration_and_image()
                    else:
                        print("Please complete fisheye correction first")

            # Start Qt event loop with hidden cursor
            return qt_run_viewer("Fisheye Correction", frame_provider, on_mouse, on_key, hide_cursor=True)
        window_name = "Fisheye Correction"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        
        # Set initial window size to match original image dimensions
        cv2.resizeWindow(window_name, self.width, self.height)
        
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.update_display()
        
        print("Fisheye Correction Tool")
        print("=" * 40)
        print("Instructions:")
        print("1. Click points along ANY visible wall edges (even just a few points help!)")
        print("2. Focus on the TOP wall - it's most visible and important")
        print("3. For side walls: click whatever small portions you can see")
        print("4. Use '+' / '-' or the mouse wheel to zoom towards the cursor (double-click LEFT/RIGHT for quick zoom)")
        print("5. Hold RIGHT mouse (or SHIFT+Left) and drag to pan, or tap W/A/S/D to nudge the view")
        print("6. Press 'n' to advance to next wall (even with few points)")
        print("7. Press Shift+Z/CTRL+Z to undo last point")
        print("8. After correction: Press 't' to toggle views, 's' to save")
        print()
        print(f"Starting with {self.walls[0].wall_name} wall - click ANY visible curved edge")
        print("(Don't worry if you can only see small portions - that's normal for fisheye!)")
        print("TIP: The top wall is most important - spend time getting many points there")
        
        wait_key = getattr(cv2, 'waitKeyEx', cv2.waitKey)

        try:
            while True:
                cv2.imshow(window_name, self.display_image)
                raw_key = wait_key(1)
                key = raw_key & 0xFF if raw_key != -1 else -1
                
                # Check for window close event
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed - exiting...")
                    break
                
                if self._key_in(key, raw_key, 'q', 'Q'):
                    print("Quitting...")
                    break
                # Handle undo (CTRL+Z, Shift+Z)
                elif key in (26, ord('z'), ord('Z')):
                    self.undo_last_point()
                elif self._key_in(key, raw_key, '+', '='):
                    cx, cy = self.cursor_pos
                    self._zoom_at_screen_point(1, cx, cy)
                    self.update_display()
                elif self._key_in(key, raw_key, '-', '_'):
                    cx, cy = self.cursor_pos
                    self._zoom_at_screen_point(-1, cx, cy)
                    self.update_display()
                elif self._key_in(key, raw_key, 's', 'S'):
                    # Save calibration and corrected image (handle before pan to avoid 's' conflict)
                    if self.corrected_image is not None:
                        self.save_calibration_and_image()
                    else:
                        print("Please complete fisheye correction first")
                elif self._handle_keyboard_pan(key, raw_key):
                    continue
                elif self._key_in(key, raw_key, 'n', 'N'):
                    # Advance to next wall
                    self.advance_to_next_wall()
                elif self._key_in(key, raw_key, 'r', 'R'):
                    # Reset current wall
                    if self.current_wall_idx < len(self.walls):
                        wall_name = self.walls[self.current_wall_idx].wall_name
                        self.walls[self.current_wall_idx].points.clear()
                        self.update_display()
                        print(f"Reset {wall_name} wall points")
                elif self._key_in(key, raw_key, 'c', 'C'):
                    # Manual calculate distortion correction
                    if not self.auto_corrected:
                        print("Calculating distortion correction...")
                        self.auto_calculate_correction()
                elif self._key_in(key, raw_key, 't', 'T'):
                    # Toggle between original and corrected view
                    if self.corrected_image is not None:
                        self.show_corrected = not self.show_corrected
                        self.update_display()
                        
                        # Resize window to match current view
                        if self.show_corrected:
                            corrected_width, corrected_height = self.fisheye_calibration.corrected_size
                            cv2.resizeWindow(window_name, corrected_width, corrected_height)
                            print("\n" + "="*50)
                            print("NOW SHOWING: CORRECTED VIEW")
                            print("="*50)
                            print("✓ Fisheye distortion removed")
                            print("✓ Curved walls now appear straight")
                            print("✓ Arena looks rectangular with visible corners")
                        else:
                            corrected_width, corrected_height = self.fisheye_calibration.corrected_size
                            cv2.resizeWindow(window_name, corrected_width, corrected_height)  # Same size for seamless toggle
                            print("\n" + "="*50)
                            print("NOW SHOWING: ORIGINAL VIEW") 
                            print("="*50)
                            print("⚠ Original fisheye distorted image (with margins)")
                            print("⚠ Walls appear curved due to fisheye lens")
                            print("⚠ Arena corners may be outside view")
                    else:
                        print("Complete fisheye correction first")
                # (save handled earlier to avoid conflict with 's' pan)
                        
        except KeyboardInterrupt:
            print("\nInterrupted by user - exiting...")
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            cv2.destroyAllWindows()
            print("Application closed.")


def main():
    parser = argparse.ArgumentParser(description="Fisheye distortion correction tool")
    parser.add_argument("image_path", help="Path to the input image")
    args = parser.parse_args()
    
    try:
        corrector = FisheyeCorrector(args.image_path)
        corrector.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())