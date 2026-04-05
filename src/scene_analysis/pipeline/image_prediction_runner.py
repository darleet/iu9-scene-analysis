from __future__ import annotations

from pathlib import Path

from scene_analysis.io.artifact_writer import ArtifactWriter
from scene_analysis.io.heatmap_prediction_writer import HeatmapPredictionWriter
from scene_analysis.io.image_reader import ImageDirectoryReader
from scene_analysis.pipeline.base import SceneAnalysisPipeline
from scene_analysis.types import FrameData


class ImagePredictionRunner:
    def __init__(
        self,
        pipeline: SceneAnalysisPipeline,
        image_reader: ImageDirectoryReader,
        prediction_writer: HeatmapPredictionWriter,
        artifact_writer: ArtifactWriter | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.image_reader = image_reader
        self.prediction_writer = prediction_writer
        self.artifact_writer = artifact_writer

    def run(self, max_images: int | None = None) -> int:
        """Сгенерировать predictions для всех найденных изображений"""
        processed_samples = 0
        for frame in self.image_reader.read_frames(max_images=max_images):
            result = self.pipeline.process_frame(frame)
            if result.obstacle_heatmap.heatmap is None:
                raise RuntimeError(
                    f"Obstacle heatmap is unavailable for sample '{self._sample_id_from_frame(frame)}'"
                )

            self.prediction_writer.save_prediction(
                sample_id=self._sample_id_from_frame(frame),
                heatmap=result.obstacle_heatmap.heatmap,
            )
            if self.artifact_writer is not None:
                self.artifact_writer.append_result_with_id(
                    artifact_id=self._sample_id_from_frame(frame),
                    result=result,
                )
            processed_samples += 1

        return processed_samples

    @staticmethod
    def _sample_id_from_frame(frame: FrameData) -> str:
        if frame.source_path is None:
            return f"sample_{frame.frame_index:06d}"
        return Path(frame.source_path).stem
