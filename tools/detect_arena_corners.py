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
        # Middle mouse pan
        if event == cv2.EVENT_MBUTTONDOWN:
            self.is_panning = True
            self.last_mouse_pos = (x, y)
            return
        elif event == cv2.EVENT_MBUTTONUP:
            self.is_panning = False
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
        """Mark a point on the current wall (need 2 points per wall)"""
        if self.current_wall_index < len(self.wall_sequence):
            wall_name = self.wall_sequence[self.current_wall_index]
            
            # Initialize wall points list if needed
            if wall_name not in self.wall_points:
                self.wall_points[wall_name] = []
                
            # Only allow 2 points per wall
            if len(self.wall_points[wall_name]) < 2:
                self.wall_points[wall_name].append((x, y))
                num_points = len(self.wall_points[wall_name])
                
                print(f"Added point ({x}, {y}) to {wall_name} wall [{num_points}/2 points]")
                
                # If we have 2 points, create the extended line
                if num_points == 2:
                    self.create_wall_line(wall_name)
                    print(f"✓ {wall_name} wall line created! Press 'n' for next wall")
                    
                self.update_display()
            else:
                print(f"{wall_name} wall already has 2 points. Press 'n' for next wall or 'r' to reset")
                
    def create_wall_line(self, wall_name: str):
        """Create extended line from 2 wall points"""
        if wall_name not in self.wall_points or len(self.wall_points[wall_name]) != 2:
            return
            
        p1, p2 = self.wall_points[wall_name]
        
        # Calculate line direction
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Extend line to window edges
        if abs(dx) > abs(dy):  # More horizontal than vertical
            # Extend to left and right edges
            if dx != 0:
                # Calculate y values at x=0 and x=width
                slope = dy / dx
                y_at_0 = p1[1] - slope * p1[0]
                y_at_width = p1[1] + slope * (self.width - p1[0])
                
                start_point = (0, y_at_0)
                end_point = (self.width, y_at_width)
            else:
                # Vertical line
                start_point = (p1[0], 0)
                end_point = (p1[0], self.height)
        else:  # More vertical than horizontal
            # Extend to top and bottom edges
            if dy != 0:
                # Calculate x values at y=0 and y=height
                slope = dx / dy
                x_at_0 = p1[0] - slope * p1[1]
                x_at_height = p1[0] + slope * (self.height - p1[1])
                
                start_point = (x_at_0, 0)
                end_point = (x_at_height, self.height)
            else:
                # Horizontal line
                start_point = (0, p1[1])
                end_point = (self.width, p1[1])
                
        self.wall_lines[wall_name] = (start_point, end_point)
        
        # Detect corners after each wall line is created
        self.detect_corners()
        
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
        
        # Apply viewport to base image
        self.display_image = self._apply_viewport(base)

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
        x0, y0, roi_w, roi_h, view_w, view_h = self.current_view_roi
        move_x = (dx_view * roi_w) / max(1, view_w)
        move_y = (dy_view * roi_h) / max(1, view_h)
        cx, cy = self.view_center
        self.view_center = (int(round(cx - move_x)), int(round(cy - move_y)))
                       
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
        print(f"\n" + "="*50)
        print("CORNER DETECTION SUMMARY")
        print("="*50)
        print(f"Source image: {self.image_path}")
        print(f"Image dimensions: {self.width}x{self.height}")
        print(f"Wall lines created: {len(self.wall_lines)}")
        print(f"Corners detected: {len(self.detected_corners)}")
        
        if self.detected_corners:
            print(f"\nDetected corner positions:")
            for corner_name, (x, y) in self.detected_corners.items():
                print(f"  {corner_name}: ({x:.1f}, {y:.1f})")
                
        print(f"\nFiles saved:")
        print(f"  {corner_image_rel} - Visual corner detection")
        print(f"  {corner_data_rel} - Corner data for rotation correction")
        
        print(f"\nNext step:")
        print(f"  Use rotation correction tool with: {corner_data_path}")
        
    def run(self):
        """Run the arena corner detection tool"""
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
        print("4. Press 'n' to advance, CTRL+Z to undo, 's' to save corners")
        print()
        print("WALL MARKING MODE")
        print("=" * 20)
        if self.wall_sequence:
            first_wall = self.wall_sequence[0]
            print(f"Mark 2 points on the {first_wall} wall to define the wall line")
        
        try:
            while True:
                cv2.imshow(window_name, self.display_image)
                key = cv2.waitKey(1) & 0xFF
                
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
                elif key == 26:  # CTRL+Z
                    # Undo last point
                    self.undo_last_point()
                elif key == ord('s'):
                    # Save corner data
                    self.save_corner_data()
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