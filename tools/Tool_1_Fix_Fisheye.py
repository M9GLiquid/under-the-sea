#!/usr/bin/env python3
"""
Fisheye correction tool - focuses only on detecting and correcting fisheye distortion.

Usage: python Tool_1_Fix_Fisheye.py <image_path>

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


@dataclass(frozen=True)
class SnapProfile:
    """Tuning parameters for wall snapping in the fixed arena."""
    orientation: str         # 'vertical' (use grad X) or 'horizontal' (use grad Y)
    wall_side: str           # Which side of the edge holds the wall ('left','right','top','bottom')
    grad_threshold: float    # Minimum gradient magnitude to consider
    search_radius: int       # Pixels to scan perpendicular to the wall (both directions)
    color_weight: float      # Penalty multiplier for color mismatch
    color_tolerance: float   # Max acceptable LAB delta before rejecting candidate
    lab_wall: np.ndarray     # Expected LAB color on the wall side (0..255)
    lab_floor: np.ndarray    # Expected LAB color on the floor side (0..255)


class FisheyeCorrector:
    """Tool for fisheye distortion correction"""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        self.display_image = self.original_image.copy()
        self.height, self.width = self.original_image.shape[:2]
        self.original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.original_lab = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.snap_enabled: bool = True
        self.snap_sample_offset: int = 4  # Base offset, will be multiplied by 1.5 in sampling functions
        self._quit_pipeline: bool = False  # Flag to track if user wants to quit entire pipeline
        self.snap_distance_weight: float = 0.5  # Reduced distance penalty for more snapping
        self.snap_debug_mode: bool = False  # Debug visualization for snapping
        self.snap_debug_info: Optional[dict] = None  # Store debug info for visualization
        # Sample arena-specific LAB references from known regions
        self.lab_wall_top = self._ensure_lab_reference(
            self._sample_lab_region((520, 40, 820, 160)),
            np.array([162.86, 1.36, 21.72], dtype=np.float32),
        )
        self.lab_wall_side = self._ensure_lab_reference(
            self._sample_lab_region((80, 360, 200, 580)),
            np.array([158.61, 77.80, 17.52], dtype=np.float32),
        )
        self.lab_floor = self._ensure_lab_reference(
            self._sample_lab_region((520, 700, 900, 860)),
            np.array([149.81, 134.85, 1.35], dtype=np.float32),
        )
        # More lenient color tolerance for brown wall vs white floor
        tolerance = 120.0  # Increased from 90.0 to be more forgiving
        # Increased search radius and adjusted thresholds for better detection
        self.snap_profiles = {
            "left": SnapProfile("vertical", "left", grad_threshold=15.0, search_radius=60, color_weight=0.4, color_tolerance=tolerance, lab_wall=self.lab_wall_side, lab_floor=self.lab_floor),
            "right": SnapProfile("vertical", "right", grad_threshold=15.0, search_radius=60, color_weight=0.4, color_tolerance=tolerance, lab_wall=self.lab_wall_side, lab_floor=self.lab_floor),
            "top": SnapProfile("horizontal", "top", grad_threshold=12.0, search_radius=60, color_weight=0.4, color_tolerance=tolerance, lab_wall=self.lab_wall_top, lab_floor=self.lab_floor),
            "bottom": SnapProfile("horizontal", "bottom", grad_threshold=12.0, search_radius=60, color_weight=0.4, color_tolerance=tolerance, lab_wall=self.lab_wall_top, lab_floor=self.lab_floor),
        }
        self.snap_default_profile = SnapProfile("horizontal", "top", grad_threshold=12.0, search_radius=60, color_weight=0.4, color_tolerance=tolerance, lab_wall=self.lab_wall_top, lab_floor=self.lab_floor)
        self.snap_gradient_x, self.snap_gradient_y = self._compute_snap_maps(self.original_gray)
        
        # Viewport state retained for compatibility (view locked to full image)
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
        """Handle mouse events for point collection with snapping assistance."""
        # Adjust for header offset (header is 80px tall)
        header_height = 80
        if y < header_height:
            # Mouse is in header area, ignore clicks but allow cursor tracking
            if event == cv2.EVENT_MOUSEMOVE:
                self.cursor_pos = (x, y)
            return
        # Adjust y coordinate to account for header
        y_adjusted = y - header_height
        
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
            bx, by = self._screen_to_base_xy(x, y_adjusted)
            if bx is not None and by is not None:
                self.cursor_base_xy = (bx, by)
            # Refresh view to move crosshair smoothly with the mouse
            if event == cv2.EVENT_MOUSEMOVE:
                self.update_display()

        if self._handle_add_point(event, x, y_adjusted):
            return

    def _handle_pan(self, event, x, y, flags=0):
        """Panning disabled in fixed-arena workflow."""
        return False

    def _handle_zoom(self, event, x, y, flags):
        """Zoom disabled in fixed-arena workflow."""
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
                snapped_x, snapped_y = self._snap_point_to_edge(px, py)
                if (snapped_x, snapped_y) != (px, py):
                    print(f"Snapped to edge: ({px}, {py}) -> ({snapped_x}, {snapped_y})")
                    px, py = snapped_x, snapped_y
                self.walls[self.current_wall_idx].points.append((px, py))
                wall_name = self.walls[self.current_wall_idx].wall_name
                num_points = len(self.walls[self.current_wall_idx].points)
                print(f"Added point ({px}, {py}) to {wall_name} wall [{num_points} points]")
                self.update_display()
                return True
        return False
                
    def _snap_point_to_edge(self, px: int, py: int) -> Tuple[int, int]:
        """Snap a clicked location to the calibrated arena boundary.

        OFFENSIVE APPROACH: This function aggressively attacks the image boundary problem
        by using multiple layers of protection:
        1. Heavily biased search ranges that avoid image edges
        2. Massive score penalties (up to 1500 points) for candidates near boundaries
        3. Black padding detection to reject false edges
        4. Professional color validation that distinguishes brown walls from black padding

        This ensures that image boundaries (black padding vs arena content) are NEVER
        mistaken for actual wall-floor boundaries (brown wall vs white floor).
        """
        if not self.snap_enabled:
            return px, py

        profile, gradient_map = self._get_snap_profile()
        if gradient_map is None:
            return px, py

        h, w = gradient_map.shape[:2]
        px = int(np.clip(px, 0, w - 1))
        py = int(np.clip(py, 0, h - 1))

        best_point: Optional[Tuple[int, int]] = None
        best_score: float = float("-inf")
        fallback_point: Optional[Tuple[int, int]] = None
        fallback_grad: float = float("-inf")

        # Store debug information
        debug_candidates = []
        debug_search_center = (px, py)

        # PRIORITY: Check the exact click location first, then expand outward
        # This ensures we prioritize what the user actually clicked on
        border_margin = 3  # Start with small margin, use scoring to handle boundaries
        
        # First, try the exact click location (delta = 0)
        # Then expand outward in a prioritized order: closest first
        search_sequence = [0]  # Start with exact click location
        
        # Determine search range
        if profile.orientation == "vertical" and profile.wall_side == "right":
            search_start = max(-profile.search_radius, -min(80, px - border_margin))
            search_end = min(profile.search_radius // 3, 15)
        elif profile.orientation == "vertical" and profile.wall_side == "left":
            search_start = max(-profile.search_radius // 3, -15)
            search_end = min(profile.search_radius, w - px - border_margin)
        else:
            search_start = -profile.search_radius
            search_end = profile.search_radius
        
        # Build search sequence: prioritize closer distances
        # Add deltas in order: 0, ±1, ±2, ±3, ... (closest first)
        for distance in range(1, abs(search_end - search_start) + 1):
            if -distance >= search_start:
                search_sequence.append(-distance)
            if distance <= search_end:
                search_sequence.append(distance)

        for delta in search_sequence:
            if profile.orientation == "vertical":
                # For left/right walls: search horizontally (along X axis) for vertical edges
                x = px + delta
                y = py
                if x < 0 or x >= w:
                    continue
                # Only avoid the very edge pixels, not the wall areas
                if x < border_margin or x >= w - border_margin:
                    continue
                grad = float(gradient_map[y, x])
                if grad < profile.grad_threshold:
                    continue
                wall_lab, floor_lab = self._lab_samples_vertical(x, y, profile.wall_side)
            else:
                # For top/bottom walls: search vertically (along Y axis) for horizontal edges
                x = px
                y = py + delta
                if y < 0 or y >= h:
                    continue
                # Only avoid the very edge pixels, not the wall areas
                if y < border_margin or y >= h - border_margin:
                    continue
                grad = float(gradient_map[y, x])
                if grad < profile.grad_threshold:
                    continue
                wall_lab, floor_lab = self._lab_samples_horizontal(x, y, profile.wall_side)

            if wall_lab is None or floor_lab is None:
                continue
            
            # Check fallback candidate (before full validation) - only if colors look valid
            wall_L = wall_lab[0]
            floor_L = floor_lab[0]
            # Basic validation for fallback: wall should be reasonably light, floor should be light
            if wall_L >= 15.0 and floor_L >= 50.0:
                if grad > fallback_grad:
                    fallback_grad = grad
                    fallback_point = (x, y)

            # More sophisticated color validation - reject image boundaries
            wall_error = self._lab_delta(wall_lab, profile.lab_wall)
            floor_error = self._lab_delta(floor_lab, profile.lab_floor)
            contrast = self._lab_delta(wall_lab, floor_lab)

            # Check if this looks like an image boundary (one side very dark/black)
            # Image boundaries have one side with very low LAB values (black padding)
            wall_is_black = self._is_black_padding(wall_lab)
            floor_is_black = self._is_black_padding(floor_lab)

            # Reject if either side looks like black padding (likely image boundary)
            if wall_is_black or floor_is_black:
                # Store debug info for rejected candidates too
                debug_candidates.append({
                    'point': (x, y),
                    'score': float('-inf'),
                    'grad': grad,
                    'contrast': contrast,
                    'delta': delta,
                    'color_acceptable': False,
                    'wall_is_black': wall_is_black,
                    'floor_is_black': floor_is_black,
                    'wall_error': wall_error,
                    'floor_error': floor_error,
                    'wall_lab': wall_lab,
                    'floor_lab': floor_lab,
                    'rejected': True,
                    'reject_reason': 'black_padding'
                })
                continue

            # STRICT VALIDATION: Ensure both sides match their expected colors
            # This prevents floor smudges from being detected as walls
            
            # Wall side must:
            # 1. Have reasonable lightness (not too dark - walls are brown, not black)
            # 2. Match expected wall color reasonably well
            wall_L = wall_lab[0]
            wall_is_valid = wall_L >= 15.0  # Walls should be reasonably light (brown, not black)
            wall_is_valid = wall_is_valid and wall_error <= profile.color_tolerance * 1.5  # Allow some tolerance
            
            # Floor side must:
            # 1. Be light (high L value - floor is white/light)
            # 2. Match expected floor color reasonably well
            floor_L = floor_lab[0]
            floor_is_valid = floor_L >= 50.0  # Floor should be light (white/light gray)
            floor_is_valid = floor_is_valid and floor_error <= profile.color_tolerance * 1.5  # Allow some tolerance
            
            # Require BOTH sides to be valid - don't accept based on contrast alone
            # This ensures we only snap to actual wall-floor boundaries, not floor smudges
            color_acceptable = wall_is_valid and floor_is_valid
            
            # Additional check: ensure there's reasonable contrast (but not required if colors match well)
            # This helps reject cases where both sides are similar (not an edge)
            if color_acceptable and contrast < 15.0:
                # Both sides match but very low contrast - might not be a real edge
                # Still accept if colors match very well (low error)
                if wall_error > profile.color_tolerance * 0.8 or floor_error > profile.color_tolerance * 0.8:
                    color_acceptable = False

            if not color_acceptable:
                # Store debug info for rejected candidates
                debug_candidates.append({
                    'point': (x, y),
                    'score': float('-inf'),
                    'grad': grad,
                    'contrast': contrast,
                    'delta': delta,
                    'color_acceptable': False,
                    'wall_is_black': wall_is_black,
                    'floor_is_black': floor_is_black,
                    'wall_error': wall_error,
                    'floor_error': floor_error,
                    'wall_lab': wall_lab,
                    'floor_lab': floor_lab,
                    'wall_L': wall_L,
                    'floor_L': floor_L,
                    'wall_is_valid': wall_is_valid,
                    'floor_is_valid': floor_is_valid,
                    'rejected': True,
                    'reject_reason': 'color_validation_failed'
                })
                continue

            # WALL LINE VALIDATION: Check if this edge is part of a continuous wall line
            # Walls have straight-ish edges (with fisheye distortion), so neighbors should also have edges
            # Isolated smudges won't have consistent edges along the wall direction
            is_part_of_wall_line = self._verify_wall_line_continuity(x, y, profile, gradient_map, h, w)
            
            if not is_part_of_wall_line:
                # Store debug info for rejected candidates
                debug_candidates.append({
                    'point': (x, y),
                    'score': float('-inf'),
                    'grad': grad,
                    'contrast': contrast,
                    'delta': delta,
                    'color_acceptable': True,
                    'wall_is_black': wall_is_black,
                    'floor_is_black': floor_is_black,
                    'wall_error': wall_error,
                    'floor_error': floor_error,
                    'wall_lab': wall_lab,
                    'floor_lab': floor_lab,
                    'wall_L': wall_L,
                    'floor_L': floor_L,
                    'wall_is_valid': wall_is_valid,
                    'floor_is_valid': floor_is_valid,
                    'rejected': True,
                    'reject_reason': 'not_part_of_wall_line'
                })
                continue

            # OFFENSIVE: Add massive penalty for candidates near image boundaries
            # This attacks the core problem: image edges should NEVER be considered valid
            boundary_penalty = 0.0

            # Calculate distance from image boundaries
            dist_from_left = x
            dist_from_right = w - 1 - x
            dist_from_top = y
            dist_from_bottom = h - 1 - y

            # If too close to any image boundary, heavily penalize
            if dist_from_left < 20 or dist_from_right < 20 or dist_from_top < 20 or dist_from_bottom < 20:
                # Exponential penalty that gets worse as we get closer to boundary
                min_dist = min(dist_from_left, dist_from_right, dist_from_top, dist_from_bottom)
                boundary_penalty = 1000.0 * (20.0 - min_dist) / 20.0  # Up to 1000 point penalty

                # For right walls, add extra penalty if we're too far right (near image boundary)
                if profile.wall_side == "right" and dist_from_right < 30:
                    boundary_penalty += 500.0 * (30.0 - dist_from_right) / 30.0

                # For left walls, add extra penalty if we're too far left (near image boundary)
                if profile.wall_side == "left" and dist_from_left < 30:
                    boundary_penalty += 500.0 * (30.0 - dist_from_left) / 30.0

            # Prefer edges with both good gradient and good color contrast
            color_score = max(0, profile.color_tolerance - (wall_error + floor_error) * 0.5)
            if contrast > 30.0:
                color_score += contrast * 0.1  # Bonus for high contrast edges

            # PRIORITY: Heavily favor points closer to the click location
            # This ensures we use what the user actually clicked on, not a smudge 20-50 pixels away
            distance_penalty = self.snap_distance_weight * float(abs(delta))
            # Add a MASSIVE bonus for being close to the click location
            # This should heavily outweigh any gradient/color advantages of distant points
            proximity_bonus = 0.0
            if abs(delta) == 0:
                proximity_bonus = 1000.0  # Huge bonus for exact click location
            elif abs(delta) <= 5:
                proximity_bonus = 800.0   # Very large bonus for very close (within 5px)
            elif abs(delta) <= 10:
                proximity_bonus = 500.0  # Large bonus for close (within 10px)
            elif abs(delta) <= 15:
                proximity_bonus = 300.0  # Good bonus for reasonably close (within 15px)
            elif abs(delta) <= 20:
                proximity_bonus = 150.0  # Moderate bonus for somewhat close (within 20px)
            # Beyond 20 pixels, no bonus - heavily penalize distant points
            
            score = grad + color_score * 0.3 - distance_penalty - boundary_penalty + proximity_bonus

            # Store debug info for visualization
            debug_candidates.append({
                'point': (x, y),
                'score': score,
                'grad': grad,
                'contrast': contrast,
                'delta': delta,
                'boundary_penalty': boundary_penalty,
                'color_acceptable': color_acceptable,
                'wall_is_black': wall_is_black,
                'floor_is_black': floor_is_black,
                'wall_error': wall_error,
                'floor_error': floor_error,
                'wall_lab': wall_lab,
                'floor_lab': floor_lab,
                'dist_from_right': dist_from_right,
                'dist_from_left': dist_from_left
            })

            if score > best_score:
                best_score = score
                best_point = (x, y)

        # Store debug information before returning
        self.snap_debug_info = {
            'search_center': debug_search_center,
            'search_start': search_start,
            'search_end': search_end,
            'candidates': debug_candidates,
            'best_point': best_point,
            'best_score': best_score,
            'fallback_point': fallback_point,
            'profile': profile,
            'original_point': (px, py)
        }

        # PRIORITY: If we found a valid point reasonably close to click location, use it immediately
        # This prevents searching further and finding smudges 20-50 pixels away
        if best_point is not None:
            best_delta_x = abs(best_point[0] - px)
            best_delta_y = abs(best_point[1] - py)
            if profile.orientation == "vertical":
                best_delta = best_delta_x
            else:
                best_delta = best_delta_y
            
            # If best point is reasonably close (within 15 pixels), use it immediately
            # This ensures we prioritize what the user clicked on, not distant smudges
            if best_delta <= 15:
                return best_point
        
        # Otherwise, use best point if available, or fallback, or original click
        if best_point is not None:
            return best_point
        if fallback_point is not None:
            return fallback_point
        return px, py

    def _get_snap_profile(self) -> Tuple[SnapProfile, Optional[np.ndarray]]:
        """Return the tuned snap profile and corresponding gradient map."""
        wall_name: Optional[str] = None
        if 0 <= self.current_wall_idx < len(self.walls):
            wall_name = self.walls[self.current_wall_idx].wall_name.lower()
        profile = self.snap_profiles.get(wall_name, self.snap_default_profile)
        gradient_map = self.snap_gradient_x if profile.orientation == "vertical" else self.snap_gradient_y
        return profile, gradient_map

    def _verify_wall_line_continuity(self, x: int, y: int, profile, gradient_map: np.ndarray, h: int, w: int) -> bool:
        """Verify that the edge point is part of a continuous wall line.
        
        Walls have straight-ish edges (with fisheye distortion), so we check neighbors
        along the wall direction to see if they also have edges. Isolated smudges won't
        have consistent edges along the wall line.
        
        Args:
            x, y: Candidate edge point
            profile: SnapProfile with wall orientation and side
            gradient_map: Gradient map to check for edges
            h, w: Image dimensions
            
        Returns:
            True if the point appears to be part of a continuous wall line
        """
        # Sample points along the wall direction (perpendicular to the edge)
        # For vertical walls (left/right): check points above and below
        # For horizontal walls (top/bottom): check points left and right
        
        check_distance = 8  # Pixels away to check
        num_samples = 5  # Number of points to check on each side
        min_edge_points = 4  # Minimum number of points that should have edges
        
        edge_count = 0
        total_samples = 0
        
        if profile.orientation == "vertical":
            # For vertical walls: check points above and below (along Y axis)
            for offset in range(-check_distance * num_samples, check_distance * num_samples + 1, check_distance):
                if offset == 0:
                    continue  # Skip the center point (we already know it has an edge)
                check_y = y + offset
                if check_y < 0 or check_y >= h:
                    continue
                check_x = x
                total_samples += 1
                # Check if this point also has a strong edge
                if check_x >= 0 and check_x < w:
                    neighbor_grad = float(gradient_map[check_y, check_x])
                    if neighbor_grad >= profile.grad_threshold * 0.7:  # Slightly lower threshold for neighbors
                        edge_count += 1
        else:
            # For horizontal walls: check points left and right (along X axis)
            for offset in range(-check_distance * num_samples, check_distance * num_samples + 1, check_distance):
                if offset == 0:
                    continue  # Skip the center point
                check_x = x + offset
                if check_x < 0 or check_x >= w:
                    continue
                check_y = y
                total_samples += 1
                # Check if this point also has a strong edge
                if check_y >= 0 and check_y < h:
                    neighbor_grad = float(gradient_map[check_y, check_x])
                    if neighbor_grad >= profile.grad_threshold * 0.7:  # Slightly lower threshold for neighbors
                        edge_count += 1
        
        # Require at least min_edge_points neighbors to have edges
        # This ensures we're detecting a continuous wall line, not an isolated smudge
        if total_samples < 4:  # Not enough samples (near image edge)
            return True  # Allow it if we can't check (near boundaries)
        
        return edge_count >= min_edge_points
    
    @staticmethod
    def _lab_delta(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))

    @staticmethod
    def _is_black_padding(lab_color: np.ndarray) -> bool:
        """Check if a color sample looks like black padding from image boundaries."""
        if lab_color is None:
            return False

        # Black padding typically has very low L (lightness) values in LAB space
        # LAB range: L [0-100], A [-128 to +127], B [-128 to +127]
        # Black padding usually has L < 5-10, with A and B close to 0
        L, A, B = lab_color[0], lab_color[1], lab_color[2]

        # Very dark and desaturated suggests black padding
        # Black padding: L < 8, A ≈ 0, B ≈ 0 (no color tint)
        # Brown walls: L > 15-20, A > 0 (red tint), B varies (yellow/blue tint)
        if L < 8.0 and abs(A) < 2.0 and abs(B) < 2.0:
            return True

        # If it's very dark and has very low saturation, likely black padding
        # This threshold allows for some noise in the color measurements
        if L < 12.0 and abs(A) < 3.0 and abs(B) < 3.0:
            return True

        # Additional check: if it's dark and completely desaturated (A=B=0), definitely black padding
        # This catches edge cases where L might be slightly higher due to noise
        if L < 18.0 and abs(A) < 1.0 and abs(B) < 1.0:
            return True

        return False

    def _lab_patch_mean(self, cx: int, cy: int, radius: int = 1) -> np.ndarray:
        """Return the average LAB vector within a square patch centered at (cx, cy)."""
        h, w = self.original_lab.shape[:2]
        radius = max(0, int(radius))
        x0 = max(0, min(w, cx - radius))
        x1 = max(0, min(w, cx + radius + 1))
        y0 = max(0, min(h, cy - radius))
        y1 = max(0, min(h, cy + radius + 1))
        if x1 <= x0 or y1 <= y0:
            return None
        patch = self.original_lab[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        mean = patch.reshape(-1, 3).mean(axis=0)
        if np.isnan(mean).any():
            return None
        return mean.astype(np.float32)

    def _lab_samples_vertical(self, x: int, y: int, wall_side: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Sample colors on both sides of a vertical edge for left/right walls."""
        # Increased offset and radius for more reliable sampling
        offset = int(self.snap_sample_offset * 1.5)  # Increased from 4 to 6
        radius = 2  # Increased from 1 to 2 for more stable color sampling

        if wall_side == "right":
            # For right wall: wall should be to the right (higher X), floor to the left (lower X)
            wall_patch = self._lab_patch_mean(x + offset, y, radius=radius)
            floor_patch = self._lab_patch_mean(x - offset, y, radius=radius)
        else:  # default to left wall
            # For left wall: wall should be to the left (lower X), floor to the right (higher X)
            wall_patch = self._lab_patch_mean(x - offset, y, radius=radius)
            floor_patch = self._lab_patch_mean(x + offset, y, radius=radius)
        if wall_patch is None or floor_patch is None:
            return None, None
        return wall_patch, floor_patch

    def _lab_samples_horizontal(self, x: int, y: int, wall_side: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Sample colors on both sides of a horizontal edge for top/bottom walls."""
        # Increased offset and radius for more reliable sampling
        offset = int(self.snap_sample_offset * 1.5)  # Increased from 4 to 6
        radius = 2  # Increased from 1 to 2 for more stable color sampling

        if wall_side == "bottom":
            # For bottom wall: wall should be below (higher Y), floor above (lower Y)
            wall_patch = self._lab_patch_mean(x, y + offset, radius=radius)
            floor_patch = self._lab_patch_mean(x, y - offset, radius=radius)
        else:  # default to top wall
            # For top wall: wall should be above (lower Y), floor below (higher Y)
            wall_patch = self._lab_patch_mean(x, y - offset, radius=radius)
            floor_patch = self._lab_patch_mean(x, y + offset, radius=radius)
        if wall_patch is None or floor_patch is None:
            return None, None
        return wall_patch, floor_patch
                
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
        """Update the display with current points using the fixed viewport."""
        base = self._prepare_base_image()
        view = self._apply_viewport(base)
        # Draw a subtle crosshair at the cursor for consistent pointer visuals
        self._draw_crosshair(view)
        # Draw header above the image (not overlaying it)
        view_with_header = self._draw_header_overlay(view)
        self.display_image = view_with_header
    
    def _draw_header_overlay(self, image: np.ndarray) -> np.ndarray:
        """Draw header above the image (not overlaying it)."""
        h, w = image.shape[:2]
        header_height = 80
        
        # Create a new image with header space above
        canvas = np.zeros((h + header_height, w, 3), dtype=np.uint8)
        
        # Draw header bar
        cv2.rectangle(canvas, (0, 0), (w, header_height), (40, 40, 40), -1)
        cv2.rectangle(canvas, (0, header_height - 1), (w, header_height), (80, 80, 80), 1)
        
        # Determine current status
        status_text = ""
        instruction_text = ""
        
        if self.corrected_image is not None:
            status_text = "✓ Correction complete! Press 's' to save, ENTER/SPACE to continue, 'q' to quit pipeline"
        elif self.current_wall_idx < len(self.walls):
            current_wall = self.walls[self.current_wall_idx]
            wall_name = current_wall.wall_name
            point_count = len(current_wall.points)
            status_text = f"Current: {wall_name.upper()} wall | Points: {point_count} | Click on curved edges"
            instruction_text = "Press 'n' for next wall | 'c' to correct | ENTER/SPACE when done | 'q' to quit pipeline"
        else:
            status_text = "All walls complete! Press 'c' to correct, 's' to save"
            instruction_text = "ENTER/SPACE to continue | 'q' to quit pipeline"
        
        # Draw status text in header
        cv2.putText(canvas, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw instruction text if available
        if instruction_text:
            cv2.putText(canvas, instruction_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Place original image below header
        canvas[header_height:, :] = image
        
        return canvas

    def _prepare_base_image(self) -> np.ndarray:
        """Return a copy of the image base with any overlays applied."""
        if self.show_corrected and self.corrected_image is not None:
            base = self.corrected_image.copy()
            # Use the calibration margin if available, otherwise 0
            margin = self.fisheye_calibration.margin_pixels if self.fisheye_calibration else 0
        else:
            if hasattr(self, 'original_with_margins'):
                base = self.original_with_margins.copy()
                margin = self.fisheye_calibration.margin_pixels if self.fisheye_calibration else 0
            else:
                base = self.original_image.copy()
                margin = 0

        self._draw_wall_points(base, margin)

        # Add debug visualization if enabled
        if self.snap_debug_mode and self.snap_debug_info is not None:
            self._draw_snap_debug(base, margin)

        return base

    def _draw_snap_debug(self, image: np.ndarray, margin: int) -> None:
        """Draw debug visualization for snapping behavior."""
        if self.snap_debug_info is None:
            return

        debug_info = self.snap_debug_info
        search_center = debug_info['search_center']
        candidates = debug_info['candidates']
        best_point = debug_info['best_point']
        fallback_point = debug_info['fallback_point']
        profile = debug_info['profile']

        # Adjust coordinates for margin offset
        center_x, center_y = search_center[0] + margin, search_center[1] + margin

        # Draw search area based on actual search range
        if profile.orientation == "vertical":
            # Vertical search: show actual biased search range
            search_start = debug_info.get('search_start', -profile.search_radius)
            search_end = debug_info.get('search_end', profile.search_radius)
            cv2.line(image, (center_x + search_start, center_y), (center_x + search_end, center_y), (255, 255, 0), 2)
            # Draw search bounds rectangle
            cv2.rectangle(image, (center_x + search_start, center_y - 2),
                         (center_x + search_end, center_y + 2), (255, 255, 0), 1)
        else:
            # Horizontal search: vertical line
            search_radius = profile.search_radius
            cv2.line(image, (center_x, center_y - search_radius), (center_x, center_y + search_radius), (255, 255, 0), 1)
            cv2.rectangle(image, (center_x - 2, center_y - search_radius),
                         (center_x + 2, center_y + search_radius), (255, 255, 0), 1)

        # Draw all candidates with color coding
        for candidate in candidates:
            x, y = candidate['point'][0] + margin, candidate['point'][1] + margin
            score = candidate['score']
            grad = candidate['grad']
            color_acceptable = candidate['color_acceptable']
            wall_is_black = candidate.get('wall_is_black', False)
            floor_is_black = candidate.get('floor_is_black', False)
            rejected = candidate.get('rejected', False)

            # Color code based on properties - offensive approach shows boundary penalties
            boundary_penalty = candidate.get('boundary_penalty', 0)
            dist_from_right = candidate.get('dist_from_right', 999)

            if rejected and (wall_is_black or floor_is_black):
                color = (0, 0, 0)  # Black: rejected due to black padding
            elif boundary_penalty > 500:  # Heavy boundary penalty
                color = (0, 0, 139)  # Dark red: heavily penalized for being near image boundary
            elif boundary_penalty > 100:  # Moderate boundary penalty
                color = (255, 69, 0)  # Red-orange: penalized for boundary proximity
            elif color_acceptable and grad >= profile.grad_threshold:
                color = (0, 255, 0)  # Green: good candidate
            elif grad >= profile.grad_threshold:
                color = (0, 255, 255)  # Yellow: good gradient but bad color
            elif color_acceptable:
                color = (255, 165, 0)  # Orange: good color but weak gradient
            else:
                color = (128, 128, 128)  # Gray: poor candidate

            cv2.circle(image, (x, y), 2, color, -1)

            # Draw score text for top candidates
            if score > -1000:  # Only show reasonably good candidates
                if rejected and (wall_is_black or floor_is_black):
                    text = "BLACK"
                else:
                    # Show detailed info including boundary penalty
                    wall_lab = candidate.get('wall_lab', None)
                    floor_lab = candidate.get('floor_lab', None)
                    boundary_penalty = candidate.get('boundary_penalty', 0)
                    dist_from_right = candidate.get('dist_from_right', 999)

                    if boundary_penalty > 100:  # Significant boundary penalty
                        text = f"{score:.0f} BND:{boundary_penalty:.0f} R:{dist_from_right:.0f}"
                    elif wall_lab is not None and floor_lab is not None:
                        text = f"{score:.1f} W:{wall_lab[0]:.0f} F:{floor_lab[0]:.0f}"
                    else:
                        text = f"{score:.1f}"
                cv2.putText(image, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Draw best point (red circle)
        if best_point is not None:
            bx, by = best_point[0] + margin, best_point[1] + margin
            cv2.circle(image, (bx, by), 4, (0, 0, 255), -1)
            cv2.putText(image, "BEST", (bx + 5, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Draw fallback point (blue circle)
        if fallback_point is not None and fallback_point != best_point:
            fx, fy = fallback_point[0] + margin, fallback_point[1] + margin
            cv2.circle(image, (fx, fy), 3, (255, 0, 0), -1)
            cv2.putText(image, "FALLBACK", (fx + 5, fy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        # Draw info text
        search_start = debug_info.get('search_start', -profile.search_radius)
        search_end = debug_info.get('search_end', profile.search_radius)

        # Count different types of candidates for offensive debugging
        black_rejects = sum(1 for c in candidates if c.get('rejected', False) and (c.get('wall_is_black', False) or c.get('floor_is_black', False)))
        boundary_penalties = sum(1 for c in candidates if c.get('boundary_penalty', 0) > 100)
        good_candidates = sum(1 for c in candidates if not c.get('rejected', False) and c.get('color_acceptable', False) and c.get('grad', 0) >= profile.grad_threshold)

        info_text = [
            f"OFFENSIVE: {profile.wall_side} {profile.orientation} (thresh={profile.grad_threshold:.1f})",
            f"Search: {search_start} to {search_end} ({search_end - search_start}px, bias against edges)",
            f"Candidates: {len(candidates)} (good: {good_candidates}, boundary: {boundary_penalties}, black: {black_rejects})",
            f"Best score: {debug_info.get('best_score', 0):.1f}" if 'best_score' in debug_info else "No best"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(image, text, (10, 20 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

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

    def _sample_lab_region(self, rect: Tuple[int, int, int, int]) -> np.ndarray:
        """Return the average LAB color inside a rectangular region (clamped to image bounds)."""
        x0, y0, x1, y1 = rect
        h, w = self.original_lab.shape[:2]
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        if x1 <= x0 or y1 <= y0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        region = self.original_lab[y0:y1, x0:x1]
        return region.reshape(-1, 3).mean(axis=0).astype(np.float32)

    @staticmethod
    def _ensure_lab_reference(lab: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        """Fallback to provided LAB reference if sampling failed."""
        if lab is None or not np.any(lab):
            return fallback.astype(np.float32)
        return lab.astype(np.float32)

    def _compute_snap_maps(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Sobel gradient maps tailored to the fixed arena."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
        grad_x = cv2.convertScaleAbs(sobel_x)
        grad_y = cv2.convertScaleAbs(sobel_y)
        return grad_x, grad_y

    # -------------------- Viewport helpers --------------------
    def _apply_viewport(self, image: np.ndarray) -> np.ndarray:
        """Viewport is locked to the full image for the fixed arena workflow."""
        h, w = image.shape[:2]
        self.current_view_roi = (0, 0, w, h, w, h)
        return image

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
        """No-op placeholder; zooming is disabled."""
        return

    def _pan_by_pixels(self, dx_view: int, dy_view: int) -> None:
        """No-op placeholder; panning is disabled."""
        return

    def _pan_by_image_units(self, dx_image: int, dy_image: int) -> None:
        """No-op placeholder; panning is disabled."""
        return

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
        """Keyboard panning disabled."""
        return False
                        
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
            if self.fisheye_calibration is not None:
                margin = self.fisheye_calibration.margin_pixels
                expanded_size = self.fisheye_calibration.corrected_size
                self.original_with_margins = self.add_margins_to_image(self.original_image, margin, expanded_size)
            else:
                print("Warning: No calibration data available for margin adjustment")
                self.original_with_margins = self.original_image.copy()
            
            self.show_corrected = True  # Switch to corrected view after completion
            self.auto_corrected = True
            # Reset viewport for new image size
            self.view_zoom = 1.0
            if self.fisheye_calibration is not None:
                self.view_center = (expanded_size[0] // 2, expanded_size[1] // 2)
            self.update_display()  # Update display to show corrected view
            
            # Resize window to fit corrected image
            if self.fisheye_calibration is not None:
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
        
    def run(self) -> int:
        """Run the fisheye correction tool"""
        # Prefer Qt viewer (hidden OS cursor) if available
        if _HAVE_QT:
            print("Fisheye Correction Tool (Qt)")
            print("=" * 40)
            print("Instructions:")
            print("1. Click points along ANY visible wall edges (even just a few points help!)")
            print("2. Focus on the TOP wall - it's most visible and important")
            print("3. For side walls: click whatever small portions you can see")
            print("4. Snapping auto-aligns your clicks; keys: n,z,t,r,s,d,q")
            print("   (OFFENSIVE: aggressively penalizes image boundaries, heavily biased search)")
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
                    # Zoom disabled; ignore wheel input.
                    return

            def on_key(key: int) -> None:
                try:
                    ch = chr(key)
                except Exception:
                    ch = ''
                # Check for ENTER (13) or SPACEBAR (32) to proceed
                if key == 13 or key == 32:  # ENTER or SPACEBAR
                    if self.corrected_image is not None:
                        # Auto-save if correction is complete
                        self.save_calibration_and_image()
                        print("Saved and proceeding to next tool...")
                        from PyQt5 import QtWidgets
                        QtWidgets.QApplication.instance().quit()
                        return 0  # Normal exit - continue pipeline
                    else:
                        print("Please complete fisheye correction first (press 'c')")
                
                # Check for Q to quit entire pipeline
                if ch.lower() == 'q':
                    print("Quitting entire pipeline...")
                    self._quit_pipeline = True
                    from PyQt5 import QtWidgets
                    QtWidgets.QApplication.instance().quit()
                elif ch.lower() == 'z':
                    self.undo_last_point()
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
                elif ch.lower() == 'd':
                    # Toggle debug visualization
                    self.snap_debug_mode = not self.snap_debug_mode
                    self.snap_debug_info = None  # Clear previous debug info
                    self.update_display()
                    if self.snap_debug_mode:
                        print("✓ Debug visualization ENABLED - shows snapping candidates and search areas")
                        print("Move mouse to see snapping behavior, press 'd' again to disable")
                    else:
                        print("✓ Debug visualization DISABLED")

            # Start Qt event loop with hidden cursor
            qt_run_viewer("Fisheye Correction", frame_provider, on_mouse, on_key, hide_cursor=True)
            # Check if user wants to quit pipeline
            if self._quit_pipeline:
                return 2  # Special exit code to stop pipeline
            # Otherwise normal exit (user pressed ENTER/SPACE or saved)
            return 0
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
        print("4. Snapping auto-aligns your clicks—just aim near the wall/floor seam")
        print("   (OFFENSIVE: aggressively penalizes image boundaries, heavily biased search)")
        print("5. Press 'd' to toggle debug visualization (shows penalties and candidate types)")
        print("6. Press 'n' to advance to the next wall (even with few points)")
        print("7. Press Shift+Z/CTRL+Z to undo the last point")
        print("8. After correction: press 't' to toggle views, 's' to save")
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
                
                # Check for ENTER (13) or SPACEBAR (32) to proceed to next tool
                if key == 13 or key == 32:  # ENTER or SPACEBAR
                    if self.corrected_image is not None:
                        # Auto-save if correction is complete
                        self.save_calibration_and_image()
                        print("Saved and proceeding to next tool...")
                        return 0  # Normal exit - continue pipeline
                    else:
                        print("Please complete fisheye correction first (press 'c')")
                
                # Check for Q to quit entire pipeline
                if self._key_in(key, raw_key, 'q', 'Q'):
                    print("Quitting entire pipeline...")
                    return 2  # Special exit code to stop pipeline
                # Handle undo (CTRL+Z, Shift+Z)
                elif key in (26, ord('z'), ord('Z')):
                    self.undo_last_point()
                elif self._key_in(key, raw_key, 's', 'S'):
                    # Save calibration and corrected image (handle before pan to avoid 's' conflict)
                    if self.corrected_image is not None:
                        self.save_calibration_and_image()
                    else:
                        print("Please complete fisheye correction first")
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
                        if self.fisheye_calibration is not None:
                            corrected_width, corrected_height = self.fisheye_calibration.corrected_size
                            cv2.resizeWindow(window_name, corrected_width, corrected_height)

                            if self.show_corrected:
                                print("\n" + "="*50)
                                print("NOW SHOWING: CORRECTED VIEW")
                                print("="*50)
                                print("✓ Fisheye distortion removed")
                                print("✓ Curved walls now appear straight")
                                print("✓ Arena looks rectangular with visible corners")
                            else:
                                print("\n" + "="*50)
                                print("NOW SHOWING: ORIGINAL VIEW")
                                print("="*50)
                                print("⚠ Original fisheye distorted image (with margins)")
                                print("⚠ Walls appear curved due to fisheye lens")
                                print("⚠ Arena corners may be outside view")
                        else:
                            print("Warning: No calibration data available")
                    else:
                        print("Complete fisheye correction first")
                elif self._key_in(key, raw_key, 'd', 'D'):
                    # Toggle debug visualization
                    self.snap_debug_mode = not self.snap_debug_mode
                    self.snap_debug_info = None  # Clear previous debug info
                    self.update_display()
                    if self.snap_debug_mode:
                        print("✓ Debug visualization ENABLED - shows snapping candidates and search areas")
                        print("Move mouse to see snapping behavior, press 'd' again to disable")
                    else:
                        print("✓ Debug visualization DISABLED")
                # (save handled earlier to avoid conflict with 's' pan)
                        
        except KeyboardInterrupt:
            print("\nInterrupted by user - exiting pipeline...")
            cv2.destroyAllWindows()
            return 2  # Exit pipeline on interrupt
        except Exception as e:
            print(f"Error occurred: {e}")
            cv2.destroyAllWindows()
            return 1  # Error exit
        finally:
            cv2.destroyAllWindows()
            print("Application closed.")
        
        return 0  # Normal completion


def main():
    parser = argparse.ArgumentParser(description="Fisheye distortion correction tool")
    parser.add_argument("image_path", help="Path to the input image")
    args = parser.parse_args()
    
    try:
        corrector = FisheyeCorrector(args.image_path)
        exit_code = corrector.run()
        return exit_code if exit_code is not None else 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
