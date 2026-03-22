from __future__ import annotations

import math
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from scene_analysis.config import AppConfig, load_config
from scene_analysis.depth.base import create_depth_estimator
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
    depth_model: str | None,
    device: str | None,
    fp16: bool | None,
) -> AppConfig:
    if input_path is not None:
        config.input.source_path = input_path
    if output_dir is not None:
        config.output.output_dir = output_dir
    if max_frames is not None:
        config.input.max_frames = max_frames
    if sample_every_n is not None:
        config.input.sample_every_n = sample_every_n
    if depth_model is not None:
        config.depth.model = depth_model
    if device is not None:
        config.depth.device = device
    if fp16 is not None:
        config.depth.use_fp16 = fp16
    return config


def _estimate_total_frames(
    reader: VideoReader,
    max_frames: int | None,
    sample_every_n: int,
) -> int | None:
    frame_count = reader.frame_count
    if frame_count <= 0:
        return max_frames

    sampled_frame_count = math.ceil(frame_count / sample_every_n)
    if max_frames is None:
        return sampled_frame_count
    return min(sampled_frame_count, max_frames)


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
    depth_model: str | None = typer.Option(
        None,
        "--depth-model",
        help="Optional depth model override",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Optional depth device override",
    ),
    fp16: bool | None = typer.Option(
        None,
        "--fp16/--no-fp16",
        help="Enable or disable fp16 inference override",
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
            depth_model=depth_model,
            device=device,
            fp16=fp16,
        )

        setup_logging(config.runtime.log_level)
        logging_ready = True

        logger.info("Starting {} pipeline", config.app.name)
        logger.info("Input source: {}", config.input.source_path)
        logger.info("Artifacts directory: {}", config.output.output_dir)
        logger.info(
            "Depth config: enabled={} provider={} model={} requested_device={} fp16={}",
            config.depth.enabled,
            config.depth.provider,
            config.depth.model,
            config.depth.device,
            config.depth.use_fp16,
        )

        reader = VideoReader(config.input.source_path)
        preprocessor = FramePreprocessor(config.preprocessing)
        depth_estimator = create_depth_estimator(config.depth)
        obstacle_builder = DummyObstacleMapBuilder()
        dynamic_detector = DummyDynamicObjectDetector()
        pipeline = MVPSceneAnalysisPipeline(
            preprocessor=preprocessor,
            depth_estimator=depth_estimator,
            obstacle_map_builder=obstacle_builder,
            dynamic_detector=dynamic_detector,
        )
        writer = ArtifactWriter(config.output, config.depth)

        resolved_device = getattr(depth_estimator, "device", config.depth.device)
        if config.depth.enabled:
            logger.info(
                "Depth estimator ready: model={} device={}",
                config.depth.model,
                resolved_device,
            )
        else:
            logger.info("Depth estimation is disabled in config. Running preprocessing only")

        processed_frames = 0
        progress = tqdm(
            reader.read_frames(
                max_frames=config.input.max_frames,
                sample_every_n=config.input.sample_every_n,
            ),
            total=_estimate_total_frames(
                reader=reader,
                max_frames=config.input.max_frames,
                sample_every_n=config.input.sample_every_n,
            ),
            desc="Processing video",
            unit="frame",
        )
        for frame in progress:
            result = pipeline.process_frame(frame)
            writer.save_original(frame)
            writer.save_preprocessed(frame.frame_index, result.preprocessed_image)
            if result.overlay_image is not None:
                writer.save_overlay(frame.frame_index, result.overlay_image)
            writer.append_result(result)

            processed_frames += 1
            inference_ms = result.depth.metadata.get("inference_ms")
            progress.set_postfix(
                frame=frame.frame_index,
                inference_ms=f"{float(inference_ms):.1f}" if inference_ms is not None else "n/a",
            )
            logger.debug(
                "Processed frame {} at {} with depth status {}",
                frame.frame_index,
                timestamp_to_str(frame.timestamp_ms),
                result.depth.metadata.get("status", "unknown"),
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
