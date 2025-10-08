"""Shared viewport controller for OpenCV-based interactive tools."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


@dataclass(frozen=True)
class ViewportROI:
    """Viewport region of interest description."""

    x: int
    y: int
    width: int
    height: int
    view_width: int
    view_height: int


class ViewportController:
    """Reusable zoom/pan helper that keeps cursor-focused navigation consistent."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        min_zoom: float = 1.0,
        max_zoom: float = 10.0,
        zoom_step: float = 1.25,
    ) -> None:
        self.canvas_size = (int(width), int(height))
        self.min_zoom = float(min_zoom)
        self.max_zoom = float(max_zoom)
        self.zoom_step = float(max(1.01, zoom_step))
        self.zoom: float = 1.0
        self.center: Tuple[int, int] = (int(width) // 2, int(height) // 2)
        self._current_roi: Optional[ViewportROI] = None
        self._recompute_roi()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_canvas(self, width: int, height: int) -> None:
        """Update the canvas size while preserving the current viewport."""
        self.canvas_size = (int(width), int(height))
        self._clamp_center()
        self._recompute_roi()

    def reset(self) -> None:
        """Reset zoom and pan to defaults."""
        width, height = self.canvas_size
        self.zoom = 1.0
        self.center = (width // 2, height // 2)
        self._recompute_roi()

    def apply(self, image):
        """Return a zoomed/panned view for the provided image."""
        height, width = image.shape[:2]
        if (width, height) != self.canvas_size:
            self.set_canvas(width, height)
        roi = self._recompute_roi()
        x0, y0, roi_w, roi_h = roi.x, roi.y, roi.width, roi.height
        cropped = image[y0 : y0 + roi_h, x0 : x0 + roi_w]
        if roi_w == width and roi_h == height:
            return cropped
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_NEAREST)

    def screen_to_image(self, sx: int, sy: int) -> Tuple[int, int]:
        """Convert screen coordinates into image coordinates within the ROI."""
        roi = self._current_roi or self._recompute_roi()
        bx = roi.x + (sx * roi.width) / max(1, roi.view_width)
        by = roi.y + (sy * roi.height) / max(1, roi.view_height)
        return int(round(bx)), int(round(by))

    def zoom_at(self, direction: int, sx: int, sy: int) -> None:
        """Zoom in or out while keeping the cursor position stable."""
        roi = self._current_roi or self._recompute_roi()
        pre_bx = roi.x + (sx * roi.width) / max(1, roi.view_width)
        pre_by = roi.y + (sy * roi.height) / max(1, roi.view_height)
        factor = self.zoom_step if direction > 0 else 1.0 / self.zoom_step
        new_zoom = self.zoom * factor
        self.zoom = float(max(self.min_zoom, min(self.max_zoom, new_zoom)))
        roi = self._recompute_roi()
        post_bx = roi.x + (sx * roi.width) / max(1, roi.view_width)
        post_by = roi.y + (sy * roi.height) / max(1, roi.view_height)
        dx = pre_bx - post_bx
        dy = pre_by - post_by
        cx, cy = self.center
        self.center = (int(round(cx + dx)), int(round(cy + dy)))
        self._recompute_roi()

    def zoom(self, direction: int) -> None:
        """Zoom using the viewport centre as the anchor."""
        width, height = self.canvas_size
        self.zoom_at(direction, width // 2, height // 2)

    def pan_by(self, dx_view: int, dy_view: int) -> None:
        """Pan by a delta in screen pixels, translated into image space."""
        roi = self._current_roi or self._recompute_roi()
        move_x = (dx_view * roi.width) / max(1, roi.view_width)
        move_y = (dy_view * roi.height) / max(1, roi.view_height)
        cx, cy = self.center
        self.center = (int(round(cx - move_x)), int(round(cy - move_y)))
        self._clamp_center()
        self._recompute_roi()

    def get_roi(self) -> ViewportROI:
        """Return the current ROI description."""
        return self._current_roi or self._recompute_roi()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _recompute_roi(self) -> ViewportROI:
        width, height = self.canvas_size
        self._clamp_center()
        zoom = max(self.min_zoom, min(self.max_zoom, float(self.zoom)))
        roi_w = max(1, int(round(width / zoom)))
        roi_h = max(1, int(round(height / zoom)))
        half_w = roi_w // 2
        half_h = roi_h // 2
        cx, cy = self.center
        x0 = int(max(0, min(width - roi_w, cx - half_w)))
        y0 = int(max(0, min(height - roi_h, cy - half_h)))
        roi = ViewportROI(x=x0, y=y0, width=roi_w, height=roi_h, view_width=width, view_height=height)
        self._current_roi = roi
        return roi

    def _clamp_center(self) -> None:
        width, height = self.canvas_size
        zoom = max(self.min_zoom, min(self.max_zoom, float(self.zoom)))
        roi_w = max(1, int(round(width / zoom)))
        roi_h = max(1, int(round(height / zoom)))
        min_cx = roi_w // 2
        max_cx = width - (roi_w - roi_w // 2)
        min_cy = roi_h // 2
        max_cy = height - (roi_h - roi_h // 2)
        cx, cy = self.center
        cx = int(max(min_cx, min(max_cx, cx)))
        cy = int(max(min_cy, min(max_cy, cy)))
        self.center = (cx, cy)