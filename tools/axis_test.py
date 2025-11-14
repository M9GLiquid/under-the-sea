#!/usr/bin/env python3
"""
Axis Camera Snapshot Function
Callable endpoint to fetch snapshot from Axis P1346 camera and save it.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ===== EDIT THESE VALUES =====
CAMERA_IP = "192.168.1.2"  # Your camera IP
USERNAME = "quasar"       # Your username
PASSWORD = "turtlebot"    # Your password
# ==============================


def fetch_axis_snapshot(
    output_path: str = "images/GPS-Real.png",
    camera_ip: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    resolution: Optional[str] = None
) -> np.ndarray:
    """
    Fetch a snapshot from Axis camera and save it to disk.
    
    Args:
        output_path: Path to save the image (default: "images/GPS-Real.png")
        camera_ip: Camera IP address (uses hardcoded value if None)
        username: Camera username (uses hardcoded value if None)
        password: Camera password (uses hardcoded value if None)
        resolution: Image resolution as "WIDTHxHEIGHT" (e.g., "2048x1536"). 
                   If None, requests maximum resolution without cropping.
    
    Returns:
        OpenCV image array (BGR format)
    
    Raises:
        ImportError: If requests library is not installed
        ConnectionError: If connection to camera fails
        ValueError: If image decoding fails
    """
    try:
        import requests
        from requests.auth import HTTPDigestAuth
    except ImportError:
        raise ImportError("'requests' library not found. Install with: pip install requests")
    
    # Use provided values or fall back to hardcoded defaults
    ip = camera_ip or CAMERA_IP
    user = username or USERNAME
    pwd = password or PASSWORD
    
    # Build URL with resolution parameter to avoid cropping
    # Axis cameras: resolution parameter controls size, or omit for full frame
    if resolution:
        url = f"http://{ip}/axis-cgi/jpg/image.cgi?resolution={resolution}"
    else:
        # Request full resolution - try without explicit resolution first (some cameras default to full)
        # If that doesn't work, you can specify resolution explicitly
        # Some Axis cameras crop if you specify resolution, others need it
        url = f"http://{ip}/axis-cgi/jpg/image.cgi"
    
    print(f"Fetching snapshot from {ip}...")
    print(f"URL: {url}")
    
    response = requests.get(url, auth=HTTPDigestAuth(user, pwd), timeout=10)
    response.raise_for_status()
    
    # Convert to OpenCV image
    img_array = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image from camera response")
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save image (overwrites if exists)
    cv2.imwrite(str(output_file), img)
    print(f"Success! Saved snapshot ({img.shape[1]}x{img.shape[0]}) to: {output_file}")
    
    return img


# For backwards compatibility or direct script execution
if __name__ == "__main__":
    try:
        fetch_axis_snapshot()
    except Exception as e:
        print(f"Error: {e}")
