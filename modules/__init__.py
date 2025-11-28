"""
Module loader for overlay API.

Loads overlay-api.py (hyphenated filename) and exposes GPSOverlay.
"""
from importlib import util
from pathlib import Path

__all__ = ["GPSOverlay", "overlay_api"]


def _load_overlay_api():
    path = Path(__file__).resolve().parent / "overlay-api.py"
    spec = util.spec_from_file_location("overlay_api", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load overlay-api from {path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


overlay_api = _load_overlay_api()
GPSOverlay = overlay_api.GPSOverlay
