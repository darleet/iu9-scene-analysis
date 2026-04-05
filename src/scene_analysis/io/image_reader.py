from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from scene_analysis.types import FrameData
from scene_analysis.utils import list_files_by_extension

DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)


class ImageDirectoryReader:
    def __init__(self, input_dir: Path, extension: str | None = None) -> None:
        self.input_dir = input_dir.expanduser()
        self.extension = extension.lower() if extension is not None else None

    def discover_images(self, max_images: int | None = None) -> list[Path]:
        """Найти изображения в директории"""
        if max_images is not None and max_images <= 0:
            raise ValueError("max_images must be positive when provided")
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Image input directory not found: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Image input path is not a directory: {self.input_dir}")

        if self.extension is not None:
            image_paths = list_files_by_extension(self.input_dir, self.extension)
        else:
            image_paths = []
            for extension in DEFAULT_IMAGE_EXTENSIONS:
                image_paths.extend(list_files_by_extension(self.input_dir, extension))
            image_paths = sorted(set(image_paths))

        if not image_paths:
            raise FileNotFoundError(f"No images found in directory: {self.input_dir}")
        if max_images is not None:
            return image_paths[:max_images]
        return image_paths

    def read_frames(self, max_images: int | None = None) -> Iterator[FrameData]:
        """Прочитать изображения как последовательность кадров"""
        for frame_index, image_path in enumerate(self.discover_images(max_images=max_images)):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read image file: {image_path}")

            height, width = image.shape[:2]
            yield FrameData(
                frame_index=frame_index,
                timestamp_ms=0.0,
                image=image,
                source_path=str(image_path),
                width=width,
                height=height,
            )
