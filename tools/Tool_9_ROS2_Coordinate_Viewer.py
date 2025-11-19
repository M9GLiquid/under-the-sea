#!/usr/bin/env python3
"""
Tool #9 - ROS2 Coordinate Viewer (GPS → Grid)

Purpose:
- Connect to ROS2 topic to receive live robot positions from GPS server
- Show robots on both the original GPS image and the rectified grid image side-by-side
- Display coordinate transformation: GPS coordinates → Grid cell position
- Real-time visualization of robot positions as they move

Controls:
- Automatically updates when new robot positions arrive via ROS2
- 'r': Refresh/clear display
- 's': Save snapshot of current view
- 'q' or window close: quit

Requirements:
- ROS2 must be sourced (source /opt/ros/jazzy/setup.bash)
- GPS server must be running and publishing to 'robotPositions' topic
- data/GPS-Real_fisheye_calibration.json (camera intrinsics, distortion)
- data/GPS-Real_corrected_transform.json (homographies and canvas translation)
- output/GPS-Real_corrected_rectified_oriented.png (pre-saved rectified image)
- Camera IP/credentials (or use axis_test.py config) - always uses web camera

Notes:
- Uses the same coordinate transformation pipeline as Tool 7 (manual transformation)
- Loads pre-saved rectified image (no grid overlay, same as Tool 7)
- Shows robot IDs (0-9) on both images
- Displays coordinate info: GPS (row, col) → Rectified (x, y)
- Original image updates from web camera, rectified image stays pre-saved
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Tuple, Optional, List
from pathlib import Path

import cv2
import numpy as np

# Try to import requests for web image fetching
try:
    import requests
    from requests.auth import HTTPDigestAuth
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠ Warning: requests library not available. Web image fetching disabled.")

# Add parent directory to path for ROS2 API import
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

# No overlay API needed - using same approach as Tool 7

# Import ROS2 API
ros2_api_path = os.path.join(project_root, "..", "Ros2", "api", "ros2-api.py")
if not os.path.exists(ros2_api_path):
    ros2_api_path = os.path.join(project_root, "..", "Integration-v1", "apis", "ros2-api", "ros2-api.py")

if not os.path.exists(ros2_api_path):
    print("Error: Could not find ros2-api.py")
    print("Expected locations:")
    print(f"  - {os.path.join(project_root, '..', 'Ros2', 'api', 'ros2-api.py')}")
    print(f"  - {os.path.join(project_root, '..', 'Integration-v1', 'apis', 'ros2-api', 'ros2-api.py')}")
    sys.exit(1)

# Dynamic import of ros2-api (handles hyphen in filename)
import importlib.util
spec = importlib.util.spec_from_file_location("ros2_api", ros2_api_path)
ros2_api = importlib.util.module_from_spec(spec)
sys.modules["ros2_api"] = ros2_api
spec.loader.exec_module(ros2_api)

RobotPositionAPI = ros2_api.RobotPositionAPI
SpiralRow = ros2_api.SpiralRow


# Window titles
ORIGINAL_WIN_TITLE = "GPS Original (Tool #9)"
RECTIFIED_WIN_TITLE = "Grid Rectified (Tool #9)"


def _to_abs(path: str, project_root: str) -> str:
    """Convert relative path to absolute."""
    norm = path.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(norm):
        return norm
    return os.path.join(project_root, norm)


def _load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_output_dir(project_root: str) -> str:
    """Ensure output directory exists."""
    out_dir = os.path.join(project_root, "output")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_new_camera_matrix(camera_matrix: np.ndarray, margin: int, scale_factor: float) -> np.ndarray:
    """Replicate fix_fisheye's new camera matrix adjustments."""
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
    """Map a point from the original server image to the corrected image space (same as Tool 7)."""
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
    """Apply homography transformation (same as Tool 7)."""
    v = np.array([float(pt_xy[0]), float(pt_xy[1]), 1.0], dtype=np.float64)
    q = homography @ v
    if q[2] == 0:
        return float("nan"), float("nan")
    return float(q[0] / q[2]), float(q[1] / q[2])


class ROS2CoordinateViewer:
    """Viewer for ROS2 robot coordinates showing before/after transformation (like Tool 7)."""
    
    def __init__(self, 
                 fisheye_json: str = "data/GPS-Real_fisheye_calibration.json",
                 transform_json: str = "data/GPS-Real_corrected_transform.json",
                 original_image: str = "images/GPS-Real.png",
                 topic: str = "robotPositions",
                 update_rate_hz: float = 10.0,
                 camera_ip: Optional[str] = None,
                 camera_username: Optional[str] = None,
                 camera_password: Optional[str] = None,
                 server_w: int = 2048,
                 server_h: int = 1536,
                 scale_factor: float = 0.8):
        """Initialize the ROS2 coordinate viewer (same approach as Tool 7)."""
        self.project_root = project_root
        self.output_dir = _ensure_output_dir(self.project_root)
        
        # Load calibration files (same as Tool 7)
        self.fisheye_path = _to_abs(fisheye_json, self.project_root)
        self.transform_path = _to_abs(transform_json, self.project_root)
        
        if not os.path.exists(self.fisheye_path):
            raise FileNotFoundError(f"Fisheye calibration JSON not found: {self.fisheye_path}")
        if not os.path.exists(self.transform_path):
            raise FileNotFoundError(f"Transform JSON not found: {self.transform_path}")
        
        self.fisheye = _load_json(self.fisheye_path)["fisheye_calibration"]
        self.transform = _load_json(self.transform_path)
        print(f"✓ Loaded fisheye calibration from: {self.fisheye_path}")
        print(f"✓ Loaded transform from: {self.transform_path}")
        
        # Load images (same as Tool 7)
        self.original_path = _to_abs(_load_json(self.fisheye_path).get("original_image", original_image), self.project_root)
        self.rectified_path = _to_abs(self.transform.get("rectified_image", "output/GPS-Real_corrected_rectified_oriented.png"), self.project_root)
        
        # Tool 9 always uses web camera for original image (video stream)
        self.camera_ip = camera_ip
        self.camera_username = camera_username
        self.camera_password = camera_password
        
        # Initialize video stream from web camera
        self.video_capture = None
        self.use_stream = True
        
        # Try to open video stream from camera
        if self._init_video_stream():
            print("✓ Connected to camera video stream")
            # Read first frame to initialize images
            ret, self.img_original = self.video_capture.read()
            if not ret or self.img_original is None:
                print("⚠ Failed to read from video stream, trying snapshot fallback...")
                self.img_original = self._fetch_web_image()
                if self.img_original is None:
                    print("⚠ Failed to fetch snapshot, trying local file...")
                    self.img_original = self._load_local_image()
                    self.use_stream = False
                else:
                    self.use_stream = False
            else:
                print("✓ Read first frame from video stream")
        else:
            print("⚠ Failed to open video stream, trying snapshot fallback...")
            self.img_original = self._fetch_web_image()
            if self.img_original is None:
                print("⚠ Failed to fetch snapshot, trying local file...")
                self.img_original = self._load_local_image()
            else:
                print("✓ Fetched snapshot from web camera")
            self.use_stream = False
        
        if self.img_original is None:
            raise FileNotFoundError(f"Could not load original image from stream, web, or local file")
        
        # Server (GPS) image size and calibration image size (same as Tool 7)
        # Must be initialized BEFORE image transformation
        self.server_size = (int(server_w), int(server_h))
        print(f"Using server image size: {self.server_size[0]}x{self.server_size[1]}")
        
        calib_size = self.fisheye.get("image_size", [self.img_original.shape[1], self.img_original.shape[0]])
        self.calib_size = (int(calib_size[0]), int(calib_size[1]))
        
        # Camera matrices (same as Tool 7)
        # Must be initialized BEFORE image transformation
        camera_matrix = np.array(self.fisheye["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.array(self.fisheye["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)
        margin = int(self.fisheye.get("margin_pixels", 0))
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.new_camera_matrix = _build_new_camera_matrix(camera_matrix, margin, float(scale_factor))
        
        # Homographies (same as Tool 7)
        # Must be initialized BEFORE image transformation
        self.homography_image_to_world_canvas = np.array(self.transform["homography_image_to_world_canvas"], dtype=np.float64)
        
        # Transform original image to rectified view (same pipeline as Tool 7)
        print("✓ Transforming original image to rectified view...")
        self.img_rectified = self._transform_image(self.img_original)
        print("✓ Image transformation complete")
        
        # ROS2 API
        self.topic = topic
        self.update_interval = 1.0 / update_rate_hz
        self.last_update = 0.0
        
        # Robot positions: {robot_id: {"gps": (row, col), "grid": (row, col), "spiral_row": SpiralRow}}
        self.robot_positions: Dict[int, Dict] = {}
        
        # Initialize ROS2 API
        print(f"✓ Connecting to ROS2 topic: {topic}")
        self.robot_api = RobotPositionAPI(topic=topic, min_certainty=0.25, max_speed=500.0)
        self.robot_api.start()
        print("✓ ROS2 API started")
        
        # Image refresh for stream/snapshot fetching
        self.last_image_fetch = 0.0
        self.image_fetch_interval = 0.033  # Update at ~30 FPS for stream (or every ~33ms)
    
    def _load_local_image(self) -> Optional[np.ndarray]:
        """Load image from local file."""
        if not os.path.exists(self.original_path):
            # Try to get from fisheye calibration
            fisheye_path = _to_abs("data/GPS-Real_fisheye_calibration.json", self.project_root)
            if os.path.exists(fisheye_path):
                fisheye_data = _load_json(fisheye_path)
                original_img_path = fisheye_data.get("original_image", "images/GPS-Real.png")
                self.original_path = _to_abs(original_img_path, self.project_root)
        
        return cv2.imread(self.original_path)
    
    def _init_video_stream(self) -> bool:
        """Initialize video stream from Axis camera using OpenCV VideoCapture."""
        try:
            # Get camera credentials
            camera_ip = self.camera_ip
            username = self.camera_username
            password = self.camera_password
            
            # Try to get camera config from axis_test.py if available
            if not camera_ip:
                axis_test_path = os.path.join(self.project_root, "tools", "axis_test.py")
                if os.path.exists(axis_test_path):
                    try:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("axis_test", axis_test_path)
                        axis_test = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(axis_test)
                        camera_ip = camera_ip or getattr(axis_test, 'CAMERA_IP', None)
                        username = username or getattr(axis_test, 'USERNAME', None)
                        password = password or getattr(axis_test, 'PASSWORD', None)
                    except Exception:
                        pass
            
            if not camera_ip:
                return False
            
            # Build MJPEG stream URL for Axis camera
            # Format: http://username:password@ip/axis-cgi/mjpg/video.cgi
            if username and password:
                stream_url = f"http://{username}:{password}@{camera_ip}/axis-cgi/mjpg/video.cgi"
            else:
                stream_url = f"http://{camera_ip}/axis-cgi/mjpg/video.cgi"
            
            # Open video stream
            self.video_capture = cv2.VideoCapture(stream_url)
            
            if not self.video_capture.isOpened():
                # Try alternative URL format
                if username and password:
                    stream_url = f"http://{username}:{password}@{camera_ip}/mjpg/video.mjpg"
                else:
                    stream_url = f"http://{camera_ip}/mjpg/video.mjpg"
                
                if self.video_capture.isOpened():
                    self.video_capture.release()
                
                self.video_capture = cv2.VideoCapture(stream_url)
            
            if self.video_capture.isOpened():
                # Set buffer size to minimize latency
                self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return True
            
            return False
        except Exception as e:
            print(f"⚠ Failed to initialize video stream: {e}")
            return False
    
    def _fetch_web_image(self) -> Optional[np.ndarray]:
        """Fetch single snapshot from web camera (fallback if stream fails)."""
        if not REQUESTS_AVAILABLE:
            return None
        
        try:
            # Try to get camera config from axis_test.py if available
            axis_test_path = os.path.join(self.project_root, "tools", "axis_test.py")
            camera_ip = self.camera_ip
            username = self.camera_username
            password = self.camera_password
            
            # If not provided, try to read from axis_test.py
            if not camera_ip and os.path.exists(axis_test_path):
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("axis_test", axis_test_path)
                    axis_test = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(axis_test)
                    camera_ip = camera_ip or getattr(axis_test, 'CAMERA_IP', None)
                    username = username or getattr(axis_test, 'USERNAME', None)
                    password = password or getattr(axis_test, 'PASSWORD', None)
                except Exception:
                    pass
            
            if not camera_ip:
                return None
            
            # Build URL
            url = f"http://{camera_ip}/axis-cgi/jpg/image.cgi"
            
            # Fetch image
            if username and password:
                response = requests.get(url, auth=HTTPDigestAuth(username, password), timeout=5)
            else:
                response = requests.get(url, timeout=5)
            
            response.raise_for_status()
            
            # Decode image
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            return img
        except Exception as e:
            print(f"⚠ Failed to fetch web image: {e}")
            return None
    
    def _transform_image(self, image: np.ndarray) -> np.ndarray:
        """Transform entire image from original to rectified view (same pipeline as Tool 7)."""
        # Step 1: Apply fisheye correction
        # Scale image to calibration size if needed
        if image.shape[1] != self.calib_size[0] or image.shape[0] != self.calib_size[1]:
            # Resize to calibration size
            img_scaled = cv2.resize(image, self.calib_size, interpolation=cv2.INTER_LINEAR)
        else:
            img_scaled = image
        
        # Get corrected image size from fisheye calibration
        corrected_size = tuple(self.fisheye.get("corrected_size", [img_scaled.shape[1], img_scaled.shape[0]]))
        corrected_size = (int(corrected_size[0]), int(corrected_size[1]))
        
        # Create undistortion maps
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix.astype(np.float32),
            self.dist_coeffs.astype(np.float32),
            np.eye(3, dtype=np.float32),
            self.new_camera_matrix.astype(np.float32),
            corrected_size,
            cv2.CV_16SC2
        )
        
        # Apply fisheye correction
        img_corrected = cv2.remap(img_scaled, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # Step 2: Apply homography transformation to get rectified view
        # Get canvas size from transform JSON
        canvas_size = self.transform.get("canvas_size", [corrected_size[0], corrected_size[1]])
        canvas_w = int(canvas_size[0])
        canvas_h = int(canvas_size[1])
        
        # Apply homography
        img_rectified = cv2.warpPerspective(
            img_corrected,
            self.homography_image_to_world_canvas,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        return img_rectified
    
    def _map_point(self, x: int, y: int) -> Tuple[int, int]:
        """Map GPS coordinates to rectified coordinates (same as Tool 7)."""
        xc, yc = _undistort_point_to_corrected((x, y), self.camera_matrix, self.dist_coeffs, self.new_camera_matrix, self.server_size, self.calib_size)
        xr, yr = _apply_homography((xc, yc), self.homography_image_to_world_canvas)
        return int(round(xr)), int(round(yr))
    
    def _get_robot_color(self, robot_id: int) -> Tuple[int, int, int]:
        """Get color for robot based on ID."""
        colors = [
            (0, 255, 255),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (128, 0, 128),    # Purple
            (255, 165, 0),    # Orange
            (0, 128, 255),    # Light Blue
            (128, 255, 0),    # Lime
        ]
        return colors[robot_id % len(colors)]
    
    def _update_robot_positions(self) -> bool:
        """Update robot positions from ROS2 API."""
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return False
        
        # Get all robot positions
        all_positions = self.robot_api.getPosition()  # None = all robots
        
        if not isinstance(all_positions, list):
            all_positions = []
        
        self.last_update = current_time
        
        # Update robot positions
        current_ids = set()
        for position in all_positions:
            robot_id = position.id
            current_ids.add(robot_id)
            
            # Transform GPS coordinates to rectified coordinates (same as Tool 7)
            # Use col as X and row as Y (GPS coordinate system)
            try:
                rect_x, rect_y = self._map_point(int(position.col), int(position.row))
                
                self.robot_positions[robot_id] = {
                    "gps": (int(position.row), int(position.col)),
                    "rectified": (rect_x, rect_y),
                    "spiral_row": position
                }
            except Exception as e:
                print(f"⚠ Error transforming robot {robot_id}: {e}")
                continue
        
        # Remove robots that are no longer active
        to_remove = [rid for rid in self.robot_positions.keys() if rid not in current_ids]
        for rid in to_remove:
            del self.robot_positions[rid]
        
        return len(all_positions) > 0
    
    def _draw_header_overlay(self, img: np.ndarray, title: str, robot_count: int) -> np.ndarray:
        """Draw header above the image."""
        h, w = img.shape[:2]
        header_height = 100
        
        canvas = np.zeros((h + header_height, w, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (w, header_height), (40, 40, 40), -1)
        cv2.rectangle(canvas, (0, header_height - 1), (w, header_height), (80, 80, 80), 1)
        
        status_text = f"{title} | Robots: {robot_count}"
        instruction_text = "Press 'r' to refresh | 's' to save | 'q' to quit"
        
        cv2.putText(canvas, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, instruction_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Show coordinate info
        if robot_count > 0:
            coord_text = "Format: RobotID: GPS(row,col) → Rectified(x,y)"
            cv2.putText(canvas, coord_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        
        canvas[header_height:, :] = img
        return canvas
    
    def _refresh_views(self) -> None:
        """Refresh both image views."""
        # Update image from video stream or snapshot
        current_time = time.time()
        if current_time - self.last_image_fetch >= self.image_fetch_interval:
            new_img = None
            
            # Try to read from video stream first
            if self.use_stream and self.video_capture is not None and self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if ret and frame is not None:
                    new_img = frame
                else:
                    # Stream might have disconnected, try to reconnect
                    print("⚠ Video stream disconnected, attempting to reconnect...")
                    if self._init_video_stream():
                        ret, frame = self.video_capture.read()
                        if ret and frame is not None:
                            new_img = frame
                            print("✓ Video stream reconnected")
                        else:
                            self.use_stream = False
                    else:
                        self.use_stream = False
            
            # Fallback to snapshot if stream not available
            if new_img is None and not self.use_stream:
                if REQUESTS_AVAILABLE:
                    new_img = self._fetch_web_image()
            
            # Update images if we got a new frame
            if new_img is not None:
                self.img_original = new_img
                # Transform to rectified view
                self.img_rectified = self._transform_image(self.img_original)
                self.last_image_fetch = current_time
        
        left = self.img_original.copy()
        right = self.img_rectified.copy()
        
        # Draw robots on original image (GPS coordinates) - same as Tool 7
        for robot_id, info in self.robot_positions.items():
            gps_row, gps_col = info["gps"]
            color = self._get_robot_color(robot_id)
            
            # Draw on original image
            # Scale GPS coordinates to image size (same as Tool 7)
            img_h, img_w = left.shape[:2]
            
            x_orig = int((gps_col / self.server_size[0]) * img_w)
            y_orig = int((gps_row / self.server_size[1]) * img_h)
            
            if 0 <= x_orig < img_w and 0 <= y_orig < img_h:
                # Draw circle
                cv2.circle(left, (x_orig, y_orig), 10, color, -1, cv2.LINE_AA)
                cv2.circle(left, (x_orig, y_orig), 15, color, 2, cv2.LINE_AA)
                
                # Draw robot ID number
                label = f"{robot_id}"
                cv2.putText(left, label, (x_orig + 18, y_orig + 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(left, label, (x_orig + 18, y_orig + 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                
                # Show GPS coordinates
                coord_text = f"GPS({gps_row},{gps_col})"
                text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = max(0, min(x_orig - text_size[0] // 2, img_w - text_size[0]))
                text_y = max(text_size[1] + 5, y_orig - 20)
                cv2.rectangle(left, (text_x - 5, text_y - text_size[1] - 5), 
                           (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(left, coord_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw robots on rectified image (same transformation as Tool 7)
        for robot_id, info in self.robot_positions.items():
            gps_row, gps_col = info["gps"]
            rect_x, rect_y = info["rectified"]
            color = self._get_robot_color(robot_id)
            
            # Draw on rectified image (rectified coordinates are already in image space)
            img_h, img_w = right.shape[:2]
            
            # Check if point is within image bounds
            if 0 <= rect_x < img_w and 0 <= rect_y < img_h:
                # Draw circle
                cv2.circle(right, (rect_x, rect_y), 10, color, -1, cv2.LINE_AA)
                cv2.circle(right, (rect_x, rect_y), 15, color, 2, cv2.LINE_AA)
                
                # Draw robot ID number
                label = f"{robot_id}"
                cv2.putText(right, label, (rect_x + 18, rect_y + 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(right, label, (rect_x + 18, rect_y + 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                
                # Show transformation: GPS → Rectified
                transform_text = f"GPS({gps_row},{gps_col})→Rect({rect_x},{rect_y})"
                text_size = cv2.getTextSize(transform_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = max(0, min(rect_x - text_size[0] // 2, img_w - text_size[0]))
                text_y = max(text_size[1] + 5, rect_y - 20)
                cv2.rectangle(right, (text_x - 5, text_y - text_size[1] - 5), 
                           (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(right, transform_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw headers
        left_with_header = self._draw_header_overlay(left, "GPS Original", len(self.robot_positions))
        right_with_header = self._draw_header_overlay(right, "Grid Rectified", len(self.robot_positions))
        
        cv2.imshow(ORIGINAL_WIN_TITLE, left_with_header)
        cv2.imshow(RECTIFIED_WIN_TITLE, right_with_header)
        
        # Print coordinate info to console
        if self.robot_positions:
            print("\n" + "="*60)
            print("Robot Positions:")
            for robot_id in sorted(self.robot_positions.keys()):
                info = self.robot_positions[robot_id]
                gps_row, gps_col = info["gps"]
                rect_x, rect_y = info["rectified"]
                print(f"  Robot {robot_id}: GPS({gps_row},{gps_col}) → Rectified({rect_x},{rect_y})")
            print("="*60)
    
    def run(self) -> int:
        """Run the viewer."""
        cv2.namedWindow(ORIGINAL_WIN_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow(RECTIFIED_WIN_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        
        # Set initial window sizes
        h0, w0 = self.img_original.shape[:2]
        hr, wr = self.img_rectified.shape[:2]
        cv2.resizeWindow(ORIGINAL_WIN_TITLE, min(1200, w0), min(800, h0))
        cv2.resizeWindow(RECTIFIED_WIN_TITLE, min(1200, wr), min(800, hr))
        
        self._refresh_views()
        
        print("\n" + "="*60)
        print("ROS2 Coordinate Viewer (Tool #9)")
        print("="*60)
        print("Waiting for robot positions from ROS2 topic...")
        print("Controls:")
        print("  - 'r': Refresh display")
        print("  - 's': Save snapshot")
        print("  - 'q': Quit")
        print("="*60 + "\n")
        
        try:
            while True:
                # Check if windows are closed
                if cv2.getWindowProperty(ORIGINAL_WIN_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if cv2.getWindowProperty(RECTIFIED_WIN_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break
                
                # Update robot positions
                updated = self._update_robot_positions()
                if updated:
                    self._refresh_views()
                
                # Handle keyboard input
                key = cv2.waitKey(16) & 0xFF
                
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('r'):
                    print("Refreshing display...")
                    self._refresh_views()
                elif key == ord('s'):
                    # Save snapshot
                    left = self.img_original.copy()
                    right = self.img_rectified.copy()
                    
                    # Redraw robots
                    for robot_id, info in self.robot_positions.items():
                        color = self._get_robot_color(robot_id)
                        gps_row, gps_col = info["gps"]
                        rect_x, rect_y = info["rectified"]
                        
                        # Draw on original (same as Tool 7)
                        img_h, img_w = left.shape[:2]
                        x_orig = int((gps_col / self.server_size[0]) * img_w)
                        y_orig = int((gps_row / self.server_size[1]) * img_h)
                        if 0 <= x_orig < img_w and 0 <= y_orig < img_h:
                            cv2.circle(left, (x_orig, y_orig), 8, color, -1)
                            cv2.putText(left, str(robot_id), (x_orig + 15, y_orig + 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Draw on rectified (same as Tool 7)
                        img_h, img_w = right.shape[:2]
                        if 0 <= rect_x < img_w and 0 <= rect_y < img_h:
                            cv2.circle(right, (rect_x, rect_y), 8, color, -1)
                            cv2.putText(right, str(robot_id), (rect_x + 15, rect_y + 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Combine side-by-side
                    h1, h2 = left.shape[0], right.shape[0]
                    H = max(h1, h2)
                    if h1 != H:
                        left = cv2.copyMakeBorder(left, 0, H - h1, 0, 0, cv2.BORDER_CONSTANT)
                    if h2 != H:
                        right = cv2.copyMakeBorder(right, 0, H - h2, 0, 0, cv2.BORDER_CONSTANT)
                    
                    side = np.hstack([left, right])
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_path = os.path.join(self.output_dir, f"ros2_coords_{ts}.png")
                    cv2.imwrite(out_path, side)
                    print(f"✓ Saved snapshot: {os.path.relpath(out_path, self.project_root)}")
        
        finally:
            # Cleanup
            if self.video_capture is not None:
                self.video_capture.release()
                print("✓ Video stream closed")
            if self.robot_api:
                self.robot_api.stop()
                print("✓ ROS2 API stopped")
            cv2.destroyAllWindows()
        
        return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Tool #9 - ROS2 Coordinate Viewer")
    parser.add_argument("--fisheye-json", default="data/GPS-Real_fisheye_calibration.json",
                       help="Path to fisheye calibration JSON")
    parser.add_argument("--transform-json", default="data/GPS-Real_corrected_transform.json",
                       help="Path to rectification transform JSON")
    parser.add_argument("--original-image", default="images/GPS-Real.png",
                       help="Path to original GPS image (fallback if web fetch fails)")
    parser.add_argument("--topic", default="robotPositions",
                       help="ROS2 topic name for robot positions")
    parser.add_argument("--update-rate", type=float, default=10.0,
                       help="Update rate in Hz (default: 10.0)")
    parser.add_argument("--camera-ip", type=str, default=None,
                       help="Camera IP address for web image fetching")
    parser.add_argument("--camera-username", type=str, default=None,
                       help="Camera username for authentication")
    parser.add_argument("--camera-password", type=str, default=None,
                       help="Camera password for authentication")
    parser.add_argument("--server-width", type=int, default=2048,
                       help="Server original image width (default: 2048)")
    parser.add_argument("--server-height", type=int, default=1536,
                       help="Server original image height (default: 1536)")
    parser.add_argument("--scale-factor", type=float, default=0.8,
                       help="Scale factor used in corrected camera matrix (default: 0.8)")
    args = parser.parse_args()
    
    try:
        viewer = ROS2CoordinateViewer(
            fisheye_json=args.fisheye_json,
            transform_json=args.transform_json,
            original_image=args.original_image,
            topic=args.topic,
            update_rate_hz=args.update_rate,
            camera_ip=args.camera_ip,
            camera_username=args.camera_username,
            camera_password=args.camera_password,
            server_w=args.server_width,
            server_h=args.server_height,
            scale_factor=args.scale_factor
        )
        return viewer.run()
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
