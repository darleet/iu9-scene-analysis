from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from scene_analysis.config import AppConfig, load_config
from scene_analysis.depth.base import DummyDepthEstimator
from scene_analysis.dynamic.base import DummyDynamicObjectDetector
from scene_analysis.io.artifact_writer import ArtifactWriter
from scene_analysis.io.video_reader import VideoReader
from scene_analysis.logging_setup import setup_logging
from scene_analysis.obstacle_map.base import DummyObstacleMapBuilder
from scene_analysis.pipeline.mvp_pipeline import MVPSceneAnalysisPipeline
from scene_analysis.preprocessing.frame_preprocessor import FramePreprocessor
from scene_analysis.utils import timestamp_to_str

app = typer.Typer(
    help="Scene analysis (с) Павлов Иван",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    pass


def _apply_overrides(
    config: AppConfig,
    input_path: Path | None,
    output_dir: Path | None,
    max_frames: int | None,
    sample_every_n: int | None,
) -> AppConfig:
    if input_path is not None:
        config.input.source_path = input_path
    if output_dir is not None:
        config.output.output_dir = output_dir
    if max_frames is not None:
        config.input.max_frames = max_frames
    if sample_every_n is not None:
        config.input.sample_every_n = sample_every_n
    return config


@app.command("run-video")
def run_video(
    config_path: Path = typer.Option(
        ...,
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to YAML configuration file",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        help="Optional input video path override",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Optional artifact directory override",
    ),
    max_frames: int | None = typer.Option(
        None,
        "--max-frames",
        min=1,
        help="Optional maximum number of processed frames",
    ),
    sample_every_n: int | None = typer.Option(
        None,
        "--sample-every-n",
        min=1,
        help="Optional frame sampling step",
    ),
) -> None:
    """Run scene analysis pipeline for mono camera"""
    reader: VideoReader | None = None
    writer: ArtifactWriter | None = None
    logging_ready = False

    try:
        config = load_config(config_path)
        config = _apply_overrides(
            config=config,
            input_path=input_path,
            output_dir=output_dir,
            max_frames=max_frames,
            sample_every_n=sample_every_n,
        )

        setup_logging(config.runtime.log_level)
        logging_ready = True

        logger.info("Starting {} pipeline", config.app.name)
        logger.info("Input source: {}", config.input.source_path)
        logger.info("Artifacts directory: {}", config.output.output_dir)

        reader = VideoReader(config.input.source_path)
        preprocessor = FramePreprocessor(config.preprocessing)
        depth_estimator = DummyDepthEstimator()
        obstacle_builder = DummyObstacleMapBuilder()
        dynamic_detector = DummyDynamicObjectDetector()
        pipeline = MVPSceneAnalysisPipeline(
            preprocessor=preprocessor,
            depth_estimator=depth_estimator,
            obstacle_map_builder=obstacle_builder,
            dynamic_detector=dynamic_detector,
        )
        writer = ArtifactWriter(config.output)

        processed_frames = 0
        for frame in reader.read_frames(
            max_frames=config.input.max_frames,
            sample_every_n=config.input.sample_every_n,
        ):
            result = pipeline.process_frame(frame)
            writer.save_original(frame)
            writer.save_preprocessed(frame.frame_index, result.preprocessed_image)
            if result.overlay_image is not None:
                writer.save_overlay(frame.frame_index, result.overlay_image)
            writer.append_result(result)

            processed_frames += 1
            logger.info(
                "Processed frame {} at {}",
                frame.frame_index,
                timestamp_to_str(frame.timestamp_ms),
            )

        logger.info(
            "Completed processing {} frame(s). Results saved to {}",
            processed_frames,
            config.output.output_dir,
        )
    except Exception as error:
        message = f"Pipeline execution failed: {error}"
        if logging_ready:
            logger.exception(message)
        else:
            typer.echo(message, err=True)
        raise typer.Exit(code=1) from error
    finally:
        if writer is not None:
            writer.close()
        if reader is not None:
            reader.close()
