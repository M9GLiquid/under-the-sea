"""Minimal reusable PyQt5 image viewer.

Features:
- Displays numpy BGR images (converted to RGB) in a resizable window
- Optional OS cursor hiding (uses Qt.BlankCursor)
- Emits mouse and keyboard events to provided callbacks
- Simple timer-based refresh via a frame provider callable

This is purpose-built to let existing OpenCV-based tools keep their logic
while swapping the windowing/event layer when --hide-cursor is requested.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


FrameProvider = Callable[[], np.ndarray]


class ImageWidget(QtWidgets.QLabel):
    def __init__(
        self,
        frame_provider: FrameProvider,
        on_mouse: Callable[[str, int, int, int, int, int], None],
        on_key: Callable[[int], None],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self._frame_provider = frame_provider
        self._on_mouse = on_mouse
        self._on_key = on_key
        self._pixmap_size = QtCore.QSize(0, 0)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(16)  # ~60 FPS

    def sizeHint(self) -> QtCore.QSize:  # noqa: D401 - Qt override
        return QtCore.QSize(1280, 720)

    # --- Rendering ---
    def _refresh(self) -> None:
        frame = None
        try:
            frame = self._frame_provider()
        except Exception:
            frame = None
        if frame is None:
            return
        if frame.ndim != 3 or frame.shape[2] != 3:
            return
        h, w = frame.shape[:2]
        # BGR (OpenCV) → RGB (Qt)
        rgb = frame[:, :, ::-1].copy()
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qimg)
        self._pixmap_size = pm.size()
        self.setPixmap(pm)

    # --- Mouse events ---
    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        self._on_mouse("move", e.x(), e.y(), int(e.buttons()), int(e.modifiers()), 0)
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        self._on_mouse("press", e.x(), e.y(), int(e.button()), int(e.modifiers()), 0)
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        self._on_mouse("release", e.x(), e.y(), int(e.button()), int(e.modifiers()), 0)
        super().mouseReleaseEvent(e)

    def wheelEvent(self, e: QtGui.QWheelEvent) -> None:
        delta = e.angleDelta().y()
        self._on_mouse("wheel", e.x(), e.y(), int(e.buttons()), int(e.modifiers()), int(delta))
        super().wheelEvent(e)

    # --- Keyboard ---
    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        self._on_key(e.key())
        super().keyPressEvent(e)


class ImageWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        title: str,
        frame_provider: FrameProvider,
        on_mouse: Callable[[str, int, int, int, int, int], None],
        on_key: Callable[[int], None],
        hide_cursor: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.widget = ImageWidget(frame_provider, on_mouse, on_key)
        self.setCentralWidget(self.widget)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.widget.setFocusPolicy(QtCore.Qt.StrongFocus)
        if hide_cursor:
            self.setCursor(QtCore.Qt.BlankCursor)


def run_viewer(
    title: str,
    frame_provider: FrameProvider,
    on_mouse: Callable[[str, int, int, int, int, int], None],
    on_key: Callable[[int], None],
    hide_cursor: bool = False,
) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = ImageWindow(title, frame_provider, on_mouse, on_key, hide_cursor=hide_cursor)
    win.resize(win.widget.sizeHint())
    win.show()
    return app.exec_()
