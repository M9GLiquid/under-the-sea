"""Core helpers for Tool 6 pixel-to-real-world calibration."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import math
import statistics
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple


@dataclass(frozen=True)
class RatioSample:
    """Stores one pixel-to-real-world measurement."""

    name: str
    pixel_length: float
    real_mm: float

    @property
    def real_cm(self) -> float:
        return self.real_mm / 10.0

    @property
    def cm_per_pixel(self) -> float:
        if self.pixel_length <= 0:
            raise ValueError(f"Pixel length for sample '{self.name}' must be positive")
        return self.real_cm / self.pixel_length

    @property
    def pixels_per_cm(self) -> float:
        if self.real_cm <= 0:
            raise ValueError(f"Real-world measurement for sample '{self.name}' must be positive")
        return self.pixel_length / self.real_cm

    def to_dict(self) -> Dict[str, float]:
        data = asdict(self)
        data.update(
            real_cm=self.real_cm,
            cm_per_pixel=self.cm_per_pixel,
            pixels_per_cm=self.pixels_per_cm,
        )
        return data


def bounds_to_corners(bounds: Mapping[str, float]) -> Dict[str, Tuple[float, float]]:
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


def compute_pixel_spans(corners: Mapping[str, Tuple[float, float]]) -> Dict[str, float]:
    tl = corners["top_left"]
    tr = corners["top_right"]
    br = corners["bottom_right"]
    bl = corners["bottom_left"]

    def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(math.hypot(b[0] - a[0], b[1] - a[1]))

    return {
        "top": dist(tl, tr),
        "right": dist(tr, br),
        "bottom": dist(br, bl),
        "left": dist(bl, tl),
        "diag_tl_br": dist(tl, br),
        "diag_tr_bl": dist(tr, bl),
    }


def compute_real_lengths(wall_lengths_mm: Mapping[str, float]) -> Dict[str, float]:
    required = ("top", "right", "bottom", "left")
    missing = [name for name in required if name not in wall_lengths_mm]
    if missing:
        raise ValueError(f"Missing wall measurements for: {', '.join(missing)}")

    real_lengths: Dict[str, float] = {name: float(wall_lengths_mm[name]) for name in required}
    for name, value in real_lengths.items():
        if value <= 0:
            raise ValueError(f"Wall measurement '{name}' must be positive")

    horizontal = [real_lengths['top'], real_lengths['bottom']]
    vertical = [real_lengths['left'], real_lengths['right']]
    width_mm = statistics.fmean(horizontal)
    height_mm = statistics.fmean(vertical)
    diagonal_mm = math.hypot(width_mm, height_mm)

    real_lengths["diag_tl_br"] = diagonal_mm
    real_lengths["diag_tr_bl"] = diagonal_mm
    return real_lengths


def generate_ratio_samples(
    pixel_spans: Mapping[str, float],
    real_lengths_mm: Mapping[str, float],
) -> List[RatioSample]:
    samples: List[RatioSample] = []
    for name, pixel_length in pixel_spans.items():
        if pixel_length <= 0:
            raise ValueError(f"Pixel distance for '{name}' must be positive")
        if name not in real_lengths_mm:
            raise ValueError(f"Missing real-world measurement for '{name}'")
        real_mm = float(real_lengths_mm[name])
        if real_mm <= 0:
            raise ValueError(f"Real-world measurement for '{name}' must be positive")
        samples.append(
            RatioSample(name=name, pixel_length=float(pixel_length), real_mm=real_mm)
        )
    return samples


def summarize(values: Iterable[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("Cannot summarize empty sequence")
    summary: Dict[str, float] = {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": statistics.fmean(vals),
    }
    summary["stddev"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return summary


def build_calibration_payload(
    *,
    grids_manifest: str,
    rectified_image: str | None,
    arena_bounds: Mapping[str, float],
    corner_pixels: Mapping[str, Tuple[float, float]],
    pixel_spans: Mapping[str, float],
    samples: Iterable[RatioSample],
    wall_lengths_mm: Mapping[str, float],
    real_lengths_mm: Mapping[str, float] | None = None,
    axis_mm_per_pixel: Mapping[str, float] | None = None,
    timestamp: datetime | None = None,
    assumptions: MutableMapping[str, str] | None = None,
) -> Dict[str, object]:
    base_dt = timestamp or datetime.now(timezone.utc)
    ts = base_dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    samples_list = [s.to_dict() for s in samples]
    cm_per_pixel_values = [s.cm_per_pixel for s in samples]
    pixels_per_cm_values = [s.pixels_per_cm for s in samples]

    payload: Dict[str, object] = {
        "source_grids_json": grids_manifest,
        "rectified_image": rectified_image,
        "arena_bounds": dict(arena_bounds),
        "corner_pixels": {k: [float(v[0]), float(v[1])] for k, v in corner_pixels.items()},
        "pixel_spans": {k: float(v) for k, v in pixel_spans.items()},
        "wall_lengths_mm": {k: float(v) for k, v in wall_lengths_mm.items()},
        "samples": samples_list,
        "cm_per_pixel_stats": summarize(cm_per_pixel_values),
        "pixels_per_cm_stats": summarize(pixels_per_cm_values),
        "recorded_at_utc": ts,
    }
    if real_lengths_mm is not None:
        payload["real_lengths_mm"] = {k: float(v) for k, v in real_lengths_mm.items()}
    if axis_mm_per_pixel is not None:
        axis_mm = {k: float(v) for k, v in axis_mm_per_pixel.items()}
        payload["axis_mm_per_pixel"] = axis_mm
        payload["axis_cm_per_pixel"] = {k: value / 10.0 for k, value in axis_mm.items()}
    if assumptions:
        payload["assumptions"] = dict(assumptions)
    return payload


__all__ = [
    "RatioSample",
    "bounds_to_corners",
    "compute_pixel_spans",
    "compute_real_lengths",
    "generate_ratio_samples",
    "summarize",
    "build_calibration_payload",
]
