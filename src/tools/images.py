"""Shared OpenCV helpers for the arena tools."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return image


def create_window(name: str, size: Tuple[int, int] | None = None) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    if size:
        w, h = size
        cv2.resizeWindow(name, w, h)
