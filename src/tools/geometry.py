"""Geometry helpers shared between arena tools."""
from __future__ import annotations

import math
from typing import Dict, Mapping, MutableMapping, Tuple

Corner = Tuple[float, float]

ORDERED_CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")


def bounds_to_corners(bounds: Mapping[str, float]) -> Dict[str, Corner]:
    left = float(bounds["left"])
    top = float(bounds["top"])
    right = float(bounds["right"])
    bottom = float(bounds["bottom"])
    return {
        "top_left": (left, top),
        "top_right": (right, top),
        "bottom_right": (right, bottom),
        "bottom_left": (left, bottom),
    }


def corners_to_bounds(corners: Mapping[str, Tuple[float, float]]) -> Dict[str, float]:
    tl = corners["top_left"]
    tr = corners["top_right"]
    br = corners["bottom_right"]
    bl = corners["bottom_left"]
    left = min(tl[0], bl[0])
    right = max(tr[0], br[0])
    top = min(tl[1], tr[1])
    bottom = max(bl[1], br[1])
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def distance(a: Corner, b: Corner) -> float:
    return float(math.hypot(b[0] - a[0], b[1] - a[1]))


def compute_pixel_spans(corners: Mapping[str, Corner]) -> Dict[str, float]:
    tl = corners["top_left"]
    tr = corners["top_right"]
    br = corners["bottom_right"]
    bl = corners["bottom_left"]
    return {
        "top": distance(tl, tr),
        "right": distance(tr, br),
        "bottom": distance(br, bl),
        "left": distance(bl, tl),
        "diag_tl_br": distance(tl, br),
        "diag_tr_bl": distance(tr, bl),
    }


def add_corner(bundle_section: MutableMapping[str, Corner], name: str, value: Corner) -> None:
    bundle_section[name] = (float(value[0]), float(value[1]))
