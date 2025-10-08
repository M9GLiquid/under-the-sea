"""Shared IO utilities for arena processing tools."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
BUNDLE_VERSION = 1

DEFAULT_STRUCTURE: Dict[str, Any] = {
    "version": BUNDLE_VERSION,
    "metadata": {},
    "input": {},
    "fisheye": None,
    "corners": None,
    "rectification": None,
    "grids": {
        "variants": [],
        "selected": None,
    },
    "calibration": None,
}


@dataclass(slots=True)
class ArenaBundle:
    """Wrapper for a bundle dictionary and its backing path."""

    path: Path
    data: Dict[str, Any]

    @property
    def arena_id(self) -> str:
        return self.path.stem

    def section(self, key: str, default: Any) -> Any:
        current = self.data.get(key, None)
        if current is None:
            if isinstance(default, (dict, list)):
                value = deepcopy(default)
            else:
                value = default
            self.data[key] = value
            return value
        return current


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_bundle_path(identifier: str | Path) -> Path:
    """Resolve a bundle path from a file path or arena identifier."""
    ensure_directories()
    if isinstance(identifier, Path):
        candidate = identifier
    else:
        candidate = Path(identifier)

    if candidate.suffix.lower() == ".json":
        if not candidate.is_absolute():
            candidate = DATA_DIR / candidate
        return candidate.resolve()

    # Treat as arena identifier without extension
    safe_name = candidate.stem or "arena"
    return (DATA_DIR / f"{safe_name}.json").resolve()


def load_bundle(identifier: str | Path) -> ArenaBundle:
    """Load an arena bundle, creating a default skeleton if needed."""
    bundle_path = normalize_bundle_path(identifier)
    data: Dict[str, Any]
    if bundle_path.exists():
        with bundle_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = json.loads(json.dumps(DEFAULT_STRUCTURE))
        data["metadata"]["created"] = True
    data.setdefault("version", BUNDLE_VERSION)
    bundle = ArenaBundle(path=bundle_path, data=data)
    return bundle


def save_bundle(bundle: ArenaBundle) -> None:
    ensure_directories()
    bundle.data["version"] = BUNDLE_VERSION
    bundle.path.parent.mkdir(parents=True, exist_ok=True)
    with bundle.path.open("w", encoding="utf-8") as handle:
        json.dump(bundle.data, handle, indent=2)


def arena_id_from_path(path: str | Path) -> str:
    return Path(path).stem


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def output_path(bundle: ArenaBundle, suffix: str) -> Path:
    """Return an output path using the arena id and provided suffix."""
    ensure_directories()
    suffix = suffix.lstrip("_")
    filename = f"{bundle.arena_id}_{suffix}"
    return (OUTPUT_DIR / filename).resolve()


def ensure_list_section(bundle: ArenaBundle, section_path: Iterable[str]) -> list:
    """Ensure nested list section exists and return it."""
    cursor: MutableMapping[str, Any] = bundle.data
    path_iter = list(section_path)
    for key in path_iter[:-1]:
        cursor = cursor.setdefault(key, {})  # type: ignore[assignment]
    final_key = path_iter[-1]
    return cursor.setdefault(final_key, [])  # type: ignore[return-value]


def ensure_mapping_section(bundle: ArenaBundle, section_path: Iterable[str]) -> Dict[str, Any]:
    cursor: MutableMapping[str, Any] = bundle.data
    for key in section_path:
        cursor = cursor.setdefault(key, {})  # type: ignore[assignment]
    return cursor  # type: ignore[return-value]
