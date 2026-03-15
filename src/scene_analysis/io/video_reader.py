"""Чтение кадров видео на базе OpenCV"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from scene_analysis.types import FrameData


class VideoReader:
    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path.expanduser()
        self._capture: cv2.VideoCapture | None = None
        self._fps: float = 0.0
        self._frame_count: int = 0

    def open(self) -> None:
        """Открыть видео"""
        if self._capture is not None and self._capture.isOpened():
            return

        if not self.source_path.exists():
            raise FileNotFoundError(f"Video source not found: {self.source_path}")

        capture = cv2.VideoCapture(str(self.source_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open video source: {self.source_path}")

        self._capture = capture
        self._fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def close(self) -> None:
        """Закрыть видео"""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def fps(self) -> float:
        """Вернуть FPS источника, если он доступен"""
        self.open()
        return self._fps

    @property
    def frame_count(self) -> int:
        """Вернуть общее число кадров, если оно доступно"""
        self.open()
        return self._frame_count

    def __iter__(self) -> Iterator[FrameData]:
        """Итерация по всем кадрам из источника"""
        return self.read_frames()

    def read_frames(
        self,
        max_frames: int | None = None,
        sample_every_n: int = 1,
    ) -> Iterator[FrameData]:
        """Чтение кадров из видео"""
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when provided.")
        if sample_every_n <= 0:
            raise ValueError("sample_every_n must be positive.")

        self.open()
        assert self._capture is not None

        capture = self._capture
        raw_frame_index = 0
        yielded_frames = 0
        fallback_fps = self._fps if self._fps > 0 else 30.0

        try:
            while True:
                success, image = capture.read()
                if not success:
                    break

                current_frame_index = raw_frame_index
                raw_frame_index += 1

                if current_frame_index % sample_every_n != 0:
                    continue

                timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if not np.isfinite(timestamp_ms) or (timestamp_ms <= 0.0 and current_frame_index > 0):
                    timestamp_ms = current_frame_index * 1000.0 / fallback_fps

                height, width = image.shape[:2]
                yield FrameData(
                    frame_index=current_frame_index,
                    timestamp_ms=timestamp_ms,
                    image=image,
                    source_path=str(self.source_path),
                    width=width,
                    height=height,
                )

                yielded_frames += 1
                if max_frames is not None and yielded_frames >= max_frames:
                    break
        finally:
            self.close()
