#!/usr/bin/env python3
"""
Arena corner detection tool - detects arena corners from corrected fisheye images.

Usage: python detect_arena_corners.py <corrected_image_path>

This tool:
1. Loads a fisheye-corrected image
2. Lets you mark 2 points per wall to define wall lines
3. Automatically extends lines to window edges
4. Detects arena corners from line intersections
5. Saves corner data for rotation correction tool
"""

import cv2
import numpy as np
import json
import os
import argparse
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict

# Optional Qt viewer for hidden-cursor mode
try:
    from src.tools.qt_viewer import run_viewer as qt_run_viewer
    _HAVE_QT = True
except Exception:
    _HAVE_QT = False


class ArenaCornerDetector:
    """Tool for detecting arena corners from corrected fisheye images"""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        self.display_image = self.image.copy()
        self.height, self.width = self.image.shape[:2]
        
        # Viewport state for zoom/pan
        self.view_zoom: float = 1.0
        self.view_center: Tuple[int, int] = (self.width // 2, self.height // 2)
        self.is_panning: bool = False
        self.last_mouse_pos: Tuple[int, int] = (0, 0)
        self.cursor_pos: Tuple[int, int] = (self.width // 2, self.height // 2)
        # Track cursor in base-image coordinates for drawing a stable crosshair
        self.cursor_base_xy: Tuple[int, int] = (self.width // 2, self.height // 2)
        self.pan_button: Optional[str] = None
        # Cached ROI for screen <-> image coordinate mapping
        # (x0, y0, roi_w, roi_h, view_w, view_h)
        self.current_view_roi: Optional[Tuple[int, int, int, int, int, int]] = None
        
        # Wall marking state
        self.wall_sequence = ["top", "right", "bottom", "left"]
        self.current_wall_index = 0
        self.wall_points: Dict[str, List[Tuple[int, int]]] = {}
        self.wall_lines: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        
        # Arena data
        self.detected_corners: Dict[str, Tuple[float, float]] = {}
        
        # Colors
        self.wall_colors = {
            "top": (0, 0, 255),       # Red
            "right": (255, 0, 0),     # Blue
            "bottom": (255, 255, 0),  # Cyan
            "left": (0, 255, 0)       # Green
        }
        self.corner_color = (0, 255, 255)    # Yellow for detected corners
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events: left-click mark, middle-drag pan, wheel zoom"""
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

        shift_flag = getattr(cv2, 'EVENT_FLAG_SHIFTKEY', 0)
        ctrl_flag = getattr(cv2, 'EVENT_FLAG_CTRLKEY', 0)
        alt_flag = getattr(cv2, 'EVENT_FLAG_ALTKEY', 0)
        modifier_active = any(
            flag and (flags & flag) for flag in (shift_flag, ctrl_flag, alt_flag)
        )

        # Mouse/Modifier pan
        if event == cv2.EVENT_MBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and modifier_active):
            self.is_panning = True
            if event == cv2.EVENT_MBUTTONDOWN:
                self.pan_button = 'middle'
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.pan_button = 'right'
            else:
                self.pan_button = 'left'
            self.last_mouse_pos = (x, y)
            return
        elif event in (cv2.EVENT_MBUTTONUP, cv2.EVENT_RBUTTONUP) or (event == cv2.EVENT_LBUTTONUP and self.pan_button == 'left'):
            self.is_panning = False
            self.pan_button = None
            return
        elif event == cv2.EVENT_MOUSEMOVE and self.is_panning:
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]
            self.last_mouse_pos = (x, y)
            self._pan_by_pixels(dx, dy)
            self.update_display()
            return
        
        # Mouse wheel zoom
        if event == getattr(cv2, 'EVENT_MOUSEWHEEL', -1):
            direction = 1 if flags > 0 else -1
            self._zoom_at_screen_point(direction, x, y)
            self.update_display()
            return
        
        # Left click to mark point (map through viewport)
        if event == cv2.EVENT_LBUTTONDOWN:
            bx, by = self._screen_to_base_xy(x, y)
            if bx is None:
                return
            self.mark_wall_point(int(bx), int(by))
            
    def mark_wall_point(self, x: int, y: int):
        """Mark a point on the current wall (need 2 points per wall)."""
        if self.current_wall_index >= len(self.wall_sequence):
            print("All walls already processed; press 's' to save or 'r' to reset.")
            return

        wall_name = self.wall_sequence[self.current_wall_index]
        points = self.wall_points.setdefault(wall_name, [])

        if len(points) >= 2:
            print(f"{wall_name.capitalize()} wall already has 2 points (press 'n' to advance)")
            return

        points.append((x, y))
        num_points = len(points)
        print(f"Added point ({x}, {y}) to {wall_name} wall [{num_points}/2 points]")

        if num_points == 2:
            p1, p2 = points
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            if dx == 0 and dy == 0:
                points.pop()
                print("Points are identical; need two distinct points for this wall")
                return

            if abs(dx) >= abs(dy):
                # Treat as a horizontal-ish wall, extend left/right edges
                if dx != 0:
                    slope = dy / dx
                    y_left = p1[1] - slope * p1[0]
                    y_right = p1[1] + slope * (self.width - p1[0])
                else:
                    y_left = y_right = p1[1]
                start_point = (0.0, y_left)
                end_point = (float(self.width), y_right)
            else:
                # Treat as a vertical-ish wall, extend top/bottom edges
                if dy != 0:
                    slope = dx / dy
                    x_top = p1[0] - slope * p1[1]
                    x_bottom = p1[0] + slope * (self.height - p1[1])
                else:
                    x_top = x_bottom = p1[0]
                start_point = (x_top, 0.0)
                end_point = (x_bottom, float(self.height))

            self.wall_lines[wall_name] = (start_point, end_point)
            print(f"Created {wall_name} wall line")
            self.detect_corners()

        self.update_display()
        
    def detect_corners(self):
        """Detect arena corners from wall line intersections"""
        if len(self.wall_lines) < 2:
            return
            
        self.detected_corners = {}
        
        # Define which walls intersect to form which corners
        corner_intersections = {
            "top_left": ("top", "left"),
            "top_right": ("top", "right"),
            "bottom_right": ("bottom", "right"),
            "bottom_left": ("bottom", "left")
        }
        
        for corner_name, (wall1, wall2) in corner_intersections.items():
            if wall1 in self.wall_lines and wall2 in self.wall_lines:
                intersection = self.calculate_line_intersection(
                    self.wall_lines[wall1], 
                    self.wall_lines[wall2]
                )
                if intersection:
                    self.detected_corners[corner_name] = intersection
                    
        if self.detected_corners:
            corner_names = list(self.detected_corners.keys())
            print(f"✓ Detected corners: {', '.join(corner_names)}")
            
    def calculate_line_intersection(self, line1: Tuple[Tuple[float, float], Tuple[float, float]], 
                                  line2: Tuple[Tuple[float, float], Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """Calculate intersection point of two lines"""
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        
        # Calculate line equations: ax + by + c = 0
        # Line 1: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
        a1 = y2 - y1
        b1 = -(x2 - x1)
        c1 = (x2 - x1) * y1 - (y2 - y1) * x1
        
        # Line 2: (y4-y3)x - (x4-x3)y + (x4-x3)y3 - (y4-y3)x3 = 0
        a2 = y4 - y3
        b2 = -(x4 - x3)
        c2 = (x4 - x3) * y3 - (y4 - y3) * x3
        
        # Solve system: a1*x + b1*y + c1 = 0, a2*x + b2*y + c2 = 0
        det = a1 * b2 - a2 * b1
        
        if abs(det) < 1e-8:  # Lines are parallel
            return None
            
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        
        return (x, y)
        
    def undo_last_point(self):
        """Undo the last marked point"""
        if self.current_wall_index < len(self.wall_sequence):
            wall_name = self.wall_sequence[self.current_wall_index]
            
            if wall_name in self.wall_points and self.wall_points[wall_name]:
                removed_point = self.wall_points[wall_name].pop()
                remaining = len(self.wall_points[wall_name])
                print(f"Undid point {removed_point} from {wall_name} wall [{remaining}/2 points]")
                
                # Remove wall line if we now have less than 2 points
                if remaining < 2 and wall_name in self.wall_lines:
                    del self.wall_lines[wall_name]
                    print(f"Removed {wall_name} wall line")
                    
                # Re-detect corners
                self.detect_corners()
                self.update_display()
            else:
                print(f"No points to undo on {wall_name} wall")
                
    def advance_to_next_wall(self):
        """Advance to next wall"""
        if self.current_wall_index < len(self.wall_sequence):
            wall_name = self.wall_sequence[self.current_wall_index]
            
            # Check if current wall has 2 points
            if wall_name in self.wall_points and len(self.wall_points[wall_name]) == 2:
                print(f"✓ {wall_name} wall complete")
            else:
                print(f"Skipped {wall_name} wall (not enough points)")
                
            self.current_wall_index += 1
            
            if self.current_wall_index < len(self.wall_sequence):
                next_wall = self.wall_sequence[self.current_wall_index]
                print(f"Now mark 2 points on the {next_wall} wall (or press 'n' to skip)")
            else:
                print("✓ All walls processed! Press 's' to save corner data")
                
    def draw_dashed_line(self, img, pt1, pt2, color, thickness=2, dash_length=10, alpha=1.0):
        """Draw a dashed line between two points with transparency"""
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        
        # Calculate line length and direction
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return
            
        # Normalize direction
        dx = dx / length
        dy = dy / length
        
        # Create overlay for transparency
        overlay = img.copy()
        
        # Draw dashed line on overlay
        current_length = 0
        draw = True
        
        while current_length < length:
            if draw:
                start_x = int(x1 + dx * current_length)
                start_y = int(y1 + dy * current_length)
                end_length = min(current_length + dash_length, length)
                end_x = int(x1 + dx * end_length)
                end_y = int(y1 + dy * end_length)
                cv2.line(overlay, (start_x, start_y), (end_x, end_y), color, thickness, cv2.LINE_AA)
            
            current_length += dash_length
            draw = not draw
        
        # Blend overlay with original image using alpha
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    def update_display(self):
        """Update the display with current markings"""
        base = self.image.copy()
        
        # Draw wall points (small, subtle)
        for wall_name, points in self.wall_points.items():
            color = self.wall_colors.get(wall_name, (255, 255, 255))
            
            # Draw points
            for i, point in enumerate(points):
                cv2.circle(base, point, 3, color, -1)
                cv2.circle(base, point, 5, (255, 255, 255), 1)
                           
        # Draw extended wall lines (for working view)
        for wall_name, (start, end) in self.wall_lines.items():
            color = self.wall_colors.get(wall_name, (255, 255, 255))
            cv2.line(base, 
                    (int(start[0]), int(start[1])), 
                    (int(end[0]), int(end[1])), 
                    color, 1, cv2.LINE_AA)
                    
        # Draw dashed rectangle border if all 4 corners are detected
        if len(self.detected_corners) == 4:
            # Get corners in order: TL, TR, BR, BL
            corners_order = ["top_left", "top_right", "bottom_right", "bottom_left"]
            corner_points = [self.detected_corners[name] for name in corners_order if name in self.detected_corners]
            
            if len(corner_points) == 4:
                # Draw thin red dashed rectangle border with 30% transparency
                border_color = (0, 0, 255)  # Red (BGR format)
                for i in range(4):
                    pt1 = corner_points[i]
                    pt2 = corner_points[(i + 1) % 4]
                    self.draw_dashed_line(base, pt1, pt2, border_color, thickness=2, dash_length=12, alpha=0.3)
        
        # Apply viewport to base image and draw a crosshair at cursor
        view = self._apply_viewport(base)
        self._draw_crosshair(view)
        self.display_image = view

    # -------------------- Viewport helpers --------------------
    def _apply_viewport(self, image: np.ndarray) -> np.ndarray:
        """Apply zoom/pan to the given image and cache ROI for mapping."""
        h, w = image.shape[:2]
        self._clamp_view_center(w, h)
        zoom = max(1.0, float(self.view_zoom))
        roi_w = max(1, int(round(w / zoom)))
        roi_h = max(1, int(round(h / zoom)))
        half_w = roi_w // 2
        half_h = roi_h // 2
        cx, cy = self.view_center
        x0 = int(max(0, min(w - roi_w, cx - half_w)))
        y0 = int(max(0, min(h - roi_h, cy - half_h)))
        roi = image[y0:y0 + roi_h, x0:x0 + roi_w]
        view = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)
        self.current_view_roi = (x0, y0, roi_w, roi_h, w, h)
        return view

    def _draw_crosshair(self, view_image: np.ndarray) -> None:
        """Draw a small crosshair at the current cursor position mapped into the view image."""
        if self.current_view_roi is None or view_image is None:
            return
        x0, y0, roi_w, roi_h, view_w, view_h = self.current_view_roi
        bx, by = self.cursor_base_xy
        vx = int(round((bx - x0) * float(view_w) / max(1.0, float(roi_w))))
        vy = int(round((by - y0) * float(view_h) / max(1.0, float(roi_h))))
        h, w = view_image.shape[:2]
        vx = max(0, min(w - 1, vx))
        vy = max(0, min(h - 1, vy))
        size = 10
        color = (255, 255, 255)
        thickness = 1
        cv2.line(view_image, (max(0, vx - size), vy), (min(w - 1, vx + size), vy), color, thickness, cv2.LINE_AA)
        cv2.line(view_image, (vx, max(0, vy - size)), (vx, min(h - 1, vy + size)), color, thickness, cv2.LINE_AA)

    def _clamp_view_center(self, w: int, h: int) -> None:
        """Clamp view center so the ROI remains inside the image."""
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
        """Map screen coordinates to image pixel coordinates using cached ROI."""
        if self.current_view_roi is None:
            # no viewport cache yet; use direct mapping
            return sx, sy
        x0, y0, roi_w, roi_h, view_w, view_h = self.current_view_roi
        bx = x0 + (sx * roi_w) / max(1, view_w)
        by = y0 + (sy * roi_h) / max(1, view_h)
        return int(round(bx)), int(round(by))

    def _zoom_at_screen_point(self, direction: int, sx: int, sy: int) -> None:
        """Zoom in/out keeping the cursor position stable."""
        pre_bx, pre_by = self._screen_to_base_xy(sx, sy)
        factor = 1.25 if direction > 0 else (1.0 / 1.25)
        self.view_zoom = float(max(1.0, min(10.0, self.view_zoom * factor)))
        post_bx, post_by = self._screen_to_base_xy(sx, sy)
        if pre_bx is not None and post_bx is not None:
            dx = pre_bx - post_bx
            dy = pre_by - post_by
            cx, cy = self.view_center
            self.view_center = (int(round(cx + dx)), int(round(cy + dy)))

    def _pan_by_pixels(self, dx_view: int, dy_view: int) -> None:
        """Pan viewport by a delta expressed in screen pixels."""
        if self.current_view_roi is None:
            return
        _, _, roi_w, roi_h, view_w, view_h = self.current_view_roi
        move_x = (dx_view * roi_w) / max(1, view_w)
        move_y = (dy_view * roi_h) / max(1, view_h)
        cx, cy = self.view_center
        self.view_center = (int(round(cx - move_x)), int(round(cy - move_y)))
    def _pan_by_image_units(self, dx_image: int, dy_image: int) -> None:
        """Pan the viewport by a delta in raw image coordinates."""
        if dx_image == 0 and dy_image == 0:
            return
        cx, cy = self.view_center
        self.view_center = (int(round(cx + dx_image)), int(round(cy + dy_image)))
        self.update_display()
    def _handle_keyboard_pan(self, key: int, raw_key: int) -> bool:
        """Consume keyboard presses to pan the viewport."""
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
                       
    def save_corner_data(self):
        """Save detected corner data for rotation correction"""
        if not self.detected_corners:
            print("No corners detected to save")
            print("Mark at least 2 walls to detect corners")
            return
            
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        
        # Resolve project root and ensure output directories exist
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        output_dir = os.path.join(project_root, 'output')
        data_dir = os.path.join(project_root, 'data')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Create clean documentation image (only dashed border, no lines or markers)
        doc_image = self.image.copy()
        
        if len(self.detected_corners) == 4:
            corners_order = ["top_left", "top_right", "bottom_right", "bottom_left"]
            corner_points = [self.detected_corners[name] for name in corners_order if name in self.detected_corners]
            
            if len(corner_points) == 4:
                # Draw thin red dashed rectangle border with 30% transparency
                border_color = (0, 0, 255)  # Red (BGR format)
                for i in range(4):
                    pt1 = corner_points[i]
                    pt2 = corner_points[(i + 1) % 4]
                    self.draw_dashed_line(doc_image, pt1, pt2, border_color, thickness=2, dash_length=15, alpha=0.3)
        
        # Save clean corner visualization image into output/
        corner_image_filename = f"{base_name}_corners.png"
        corner_image_path = os.path.join(output_dir, corner_image_filename)
        cv2.imwrite(corner_image_path, doc_image)
        corner_image_rel = os.path.relpath(corner_image_path, project_root)
        print(f"✓ Saved corner visualization: {corner_image_rel}")
        
        # Save corner data for rotation correction tool
        corner_data = {
            "source_image": self.image_path,
            "image_dimensions": {
                "width": self.width,
                "height": self.height
            },
            "wall_points": self.wall_points,
            "wall_lines": {
                name: {
                    "start": line[0], 
                    "end": line[1]
                } for name, line in self.wall_lines.items()
            },
            "detected_corners": self.detected_corners,
            "corner_count": len(self.detected_corners),
            "timestamp": "2025-10-03T16:00:00Z"
        }
        
        corner_data_filename = f"{base_name}_corners.json"
        corner_data_path = os.path.join(data_dir, corner_data_filename)
        with open(corner_data_path, 'w') as f:
            json.dump(corner_data, f, indent=2)
        corner_data_rel = os.path.relpath(corner_data_path, project_root)
        print(f"✓ Saved corner data: {corner_data_rel}")
        
        # Display summary
        print("\n" + "="*50)
        print("CORNER DETECTION SUMMARY")
        print("="*50)
        print(f"Source image: {self.image_path}")
        print(f"Image dimensions: {self.width}x{self.height}")
        print(f"Wall lines created: {len(self.wall_lines)}")
        print(f"Corners detected: {len(self.detected_corners)}")
        
        if self.detected_corners:
            print("\nDetected corner positions:")
            for corner_name, (x, y) in self.detected_corners.items():
                print(f"  {corner_name}: ({x:.1f}, {y:.1f})")
                
        print("\nFiles saved:")
        print(f"  {corner_image_rel} - Visual corner detection")
        print(f"  {corner_data_rel} - Corner data for rotation correction")
        
        print("\nNext step:")
        print(f"  Use rotation correction tool with: {corner_data_path}")
        
    def run(self):
        """Run the arena corner detection tool"""
        if _HAVE_QT:
            print("Arena Corner Detection (Qt)")
            print("=" * 40)
            print("Instructions:")
            print("1. Mark 2 points per wall to define each wall line")
            print("2. Lines extend to window edges automatically")
            print("3. Wheel zoom; Right/Middle drag pan; keys: n,z,s,r,q")

            def frame_provider() -> np.ndarray:
                self.update_display()
                return self.display_image

            def on_mouse(kind: str, mx: int, my: int, button_or_buttons: int, _mods: int, delta: int) -> None:
                if kind == "move":
                    self.mouse_callback(cv2.EVENT_MOUSEMOVE, mx, my, 0, None)
                elif kind == "press":
                    if button_or_buttons == 1:
                        self.mouse_callback(cv2.EVENT_LBUTTONDOWN, mx, my, 0, None)
                    elif button_or_buttons == 2:
                        self.mouse_callback(cv2.EVENT_RBUTTONDOWN, mx, my, 0, None)
                    elif button_or_buttons == 4:
                        self.mouse_callback(cv2.EVENT_MBUTTONDOWN, mx, my, 0, None)
                elif kind == "release":
                    # No-op; placement happens on press
                    pass
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
                elif ch.lower() == 'n':
                    self.advance_to_next_wall()
                elif ch.lower() == 'z':
                    self.undo_last_point()
                elif ch.lower() == 's':
                    self.save_corner_data()
                elif ch.lower() == 'r':
                    if self.current_wall_index < len(self.wall_sequence):
                        wall_name = self.wall_sequence[self.current_wall_index]
                        if wall_name in self.wall_points:
                            self.wall_points[wall_name].clear()
                            if wall_name in self.wall_lines:
                                del self.wall_lines[wall_name]
                            print(f"Reset {wall_name} wall points")
                            self.detect_corners()
                            self.update_display()

            return qt_run_viewer("Arena Corner Detection", frame_provider, on_mouse, on_key, hide_cursor=True)
        window_name = "Arena Corner Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, self.width, self.height)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.update_display()
        
        print("Arena Corner Detection Tool")
        print("=" * 40)
        print("Instructions:")
        print("1. Mark 2 points per wall to define each wall line")
        print("2. Lines extend to window edges automatically")
        print("3. Corners detected from line intersections")
        print("4. Use '+' / '-' or the mouse wheel to zoom towards the cursor (double-click LEFT/RIGHT for quick zoom)")
        print("5. Hold RIGHT mouse (or SHIFT+Left) and drag to pan, or tap W/A/S/D to nudge the view")
        print("6. Press 'n' to advance, Shift+Z/CTRL+Z to undo, 's' to save corners")
        print()
        print("WALL MARKING MODE")
        print("=" * 20)
        if self.wall_sequence:
            first_wall = self.wall_sequence[0]
            print(f"Mark 2 points on the {first_wall} wall to define the wall line")
        
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
                
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('n'):
                    # Advance to next wall
                    self.advance_to_next_wall()
                elif key in (ord('+'), ord('=')):
                    cx, cy = self.cursor_pos
                    self._zoom_at_screen_point(1, cx, cy)
                    self.update_display()
                elif key in (ord('-'), ord('_')):
                    cx, cy = self.cursor_pos
                    self._zoom_at_screen_point(-1, cx, cy)
                    self.update_display()
                elif key == ord('s'):
                    # Save corner data (handle before pan to avoid 's' conflict)
                    self.save_corner_data()
                elif self._handle_keyboard_pan(key, raw_key):
                    continue
                elif key in (26, ord('z'), ord('Z')):  # CTRL+Z or Shift+Z
                    # Undo last point
                    self.undo_last_point()
                elif key == ord('r'):
                    # Reset current wall
                    if self.current_wall_index < len(self.wall_sequence):
                        wall_name = self.wall_sequence[self.current_wall_index]
                        if wall_name in self.wall_points:
                            self.wall_points[wall_name].clear()
                            if wall_name in self.wall_lines:
                                del self.wall_lines[wall_name]
                            print(f"Reset {wall_name} wall points")
                            self.detect_corners()  # Re-detect corners
                            self.update_display()
                            
        except KeyboardInterrupt:
            print("\nInterrupted by user - exiting...")
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            cv2.destroyAllWindows()
            print("Application closed.")


def main():
    parser = argparse.ArgumentParser(description="Arena corner detection tool")
    parser.add_argument("image_path", help="Path to the corrected fisheye image")
    args = parser.parse_args()
    
    try:
        detector = ArenaCornerDetector(args.image_path)
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())