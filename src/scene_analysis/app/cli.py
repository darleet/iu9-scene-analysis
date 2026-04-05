from __future__ import annotations

import math
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from scene_analysis.config import AppConfig, load_config
from scene_analysis.depth.base import create_depth_estimator
from scene_analysis.evaluation.runner import EvaluationRunner
from scene_analysis.io.artifact_writer import ArtifactWriter
from scene_analysis.io.heatmap_prediction_writer import HeatmapPredictionWriter
from scene_analysis.io.image_reader import ImageDirectoryReader
from scene_analysis.io.video_reader import VideoReader
from scene_analysis.logging_setup import setup_logging
from scene_analysis.obstacle_map.base import create_obstacle_heatmap_builder
from scene_analysis.pipeline.image_prediction_runner import ImagePredictionRunner
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


def _apply_run_video_overrides(
    config: AppConfig,
    input_path: Path | None,
    output_dir: Path | None,
    max_frames: int | None,
    sample_every_n: int | None,
    depth_model: str | None,
    device: str | None,
    fp16: bool | None,
    suppression_strength: float | None,
    bottom_strip_ratio: float | None,
    gamma: float | None,
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
    if suppression_strength is not None:
        config.obstacle_heatmap.road_suppression.suppression_strength = suppression_strength
    if bottom_strip_ratio is not None:
        config.obstacle_heatmap.road_suppression.bottom_strip_ratio = bottom_strip_ratio
    if gamma is not None:
        config.obstacle_heatmap.near_score.gamma = gamma
    return config


def _apply_evaluation_overrides(
    config: AppConfig,
    dataset_root: Path | None,
    images_dir: Path | None,
    predictions_dir: Path | None,
    output_dir: Path | None,
) -> AppConfig:
    if dataset_root is not None:
        config.evaluation.dataset.root_dir = dataset_root
    if images_dir is not None:
        config.evaluation.dataset.images_dir = images_dir
    if predictions_dir is not None:
        config.evaluation.dataset.predictions_dir = predictions_dir
    if output_dir is not None:
        config.evaluation.outputs.output_dir = output_dir
    return config


def _apply_image_prediction_overrides(
    config: AppConfig,
    keep_preprocessing_roi: bool,
) -> AppConfig:
    if not keep_preprocessing_roi:
        config.preprocessing.roi.enabled = False
    return config


def _resolve_dataset_path(root_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else root_dir / path


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
    suppression_strength: float | None = typer.Option(
        None,
        "--suppression-strength",
        min=0.0,
        max=1.0,
        help="Optional obstacle heatmap road suppression strength override",
    ),
    bottom_strip_ratio: float | None = typer.Option(
        None,
        "--bottom-strip-ratio",
        min=0.0,
        max=1.0,
        help="Optional obstacle heatmap bottom strip ratio override",
    ),
    gamma: float | None = typer.Option(
        None,
        "--gamma",
        min=0.0001,
        help="Optional obstacle heatmap near-score gamma override",
    ),
) -> None:
    """Run scene analysis pipeline for mono camera."""
    reader: VideoReader | None = None
    writer: ArtifactWriter | None = None
    logging_ready = False

    try:
        config = load_config(config_path)
        config = _apply_run_video_overrides(
            config=config,
            input_path=input_path,
            output_dir=output_dir,
            max_frames=max_frames,
            sample_every_n=sample_every_n,
            depth_model=depth_model,
            device=device,
            fp16=fp16,
            suppression_strength=suppression_strength,
            bottom_strip_ratio=bottom_strip_ratio,
            gamma=gamma,
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
        logger.info(
            "Obstacle heatmap config: enabled={} mode={} suppression_strength={} bottom_strip_ratio={} gamma={}",
            config.obstacle_heatmap.enabled,
            config.obstacle_heatmap.road_suppression.mode,
            config.obstacle_heatmap.road_suppression.suppression_strength,
            config.obstacle_heatmap.road_suppression.bottom_strip_ratio,
            config.obstacle_heatmap.near_score.gamma,
        )

        reader = VideoReader(config.input.source_path)
        preprocessor = FramePreprocessor(config.preprocessing)
        depth_estimator = create_depth_estimator(config.depth)
        obstacle_builder = create_obstacle_heatmap_builder(config.obstacle_heatmap)
        pipeline = MVPSceneAnalysisPipeline(
            preprocessor=preprocessor,
            depth_estimator=depth_estimator,
            obstacle_heatmap_builder=obstacle_builder,
        )
        writer = ArtifactWriter(config.output, config.depth, config.obstacle_heatmap)

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
            writer.append_result(result)

            processed_frames += 1
            inference_ms = result.depth.metadata.get("inference_ms")
            progress.set_postfix(
                frame=frame.frame_index,
                inference_ms=f"{float(inference_ms):.1f}" if inference_ms is not None else "n/a",
            )
            logger.debug(
                "Processed frame {} at {} with depth status {} and heatmap status {}",
                frame.frame_index,
                timestamp_to_str(frame.timestamp_ms),
                result.depth.metadata.get("status", "unknown"),
                result.obstacle_heatmap.metadata.get("status", "unknown"),
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


@app.command("evaluate-heatmap")
def evaluate_heatmap(
    config_path: Path = typer.Option(
        ...,
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to YAML configuration file",
    ),
    dataset_root: Path | None = typer.Option(
        None,
        "--dataset-root",
        help="Optional evaluation dataset root override",
    ),
    predictions_dir: Path | None = typer.Option(
        None,
        "--predictions-dir",
        help="Optional predictions directory override",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Optional evaluation output directory override",
    ),
) -> None:
    """Evaluate already saved obstacle heatmap predictions against local ground truth."""
    logging_ready = False

    try:
        config = load_config(config_path)
        config = _apply_evaluation_overrides(
            config=config,
            dataset_root=dataset_root,
            images_dir=None,
            predictions_dir=predictions_dir,
            output_dir=output_dir,
        )

        setup_logging(config.runtime.log_level)
        logging_ready = True

        logger.info("Starting obstacle heatmap evaluation")
        logger.info("Evaluation dataset: {}", config.evaluation.dataset.name)
        logger.info("Evaluation root: {}", config.evaluation.dataset.root_dir)
        logger.info("Evaluation predictions: {}", config.evaluation.dataset.predictions_dir)
        logger.info("Evaluation outputs: {}", config.evaluation.outputs.output_dir)

        runner = EvaluationRunner(config.evaluation)
        summary = runner.run()
        ap_text = "n/a" if math.isnan(summary.average_precision) else f"{summary.average_precision:.6f}"
        logger.info(
            "Evaluation completed: dataset={} samples={} valid_samples={} global_AP={}",
            summary.dataset_name,
            summary.num_samples,
            summary.num_valid_samples,
            ap_text,
        )
    except Exception as error:
        message = f"Evaluation failed: {error}"
        if logging_ready:
            logger.exception(message)
        else:
            typer.echo(message, err=True)
        raise typer.Exit(code=1) from error


@app.command("generate-predictions")
def generate_predictions(
    config_path: Path = typer.Option(
        ...,
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to YAML configuration file",
    ),
    dataset_root: Path | None = typer.Option(
        None,
        "--dataset-root",
        help="Optional evaluation dataset root override",
    ),
    images_dir: Path | None = typer.Option(
        None,
        "--images-dir",
        help="Optional input images directory override",
    ),
    predictions_dir: Path | None = typer.Option(
        None,
        "--predictions-dir",
        help="Optional predictions directory override",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Optional artifact output directory override for readable images and jsonl",
    ),
    keep_preprocessing_roi: bool = typer.Option(
        False,
        "--keep-preprocessing-roi/--disable-preprocessing-roi",
        help="Keep video-style preprocessing ROI for image batches. By default ROI is disabled to process full-frame dataset images.",
    ),
    max_images: int | None = typer.Option(
        None,
        "--max-images",
        min=1,
        help="Optional maximum number of processed images",
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
    suppression_strength: float | None = typer.Option(
        None,
        "--suppression-strength",
        min=0.0,
        max=1.0,
        help="Optional obstacle heatmap road suppression strength override",
    ),
    bottom_strip_ratio: float | None = typer.Option(
        None,
        "--bottom-strip-ratio",
        min=0.0,
        max=1.0,
        help="Optional obstacle heatmap bottom strip ratio override",
    ),
    gamma: float | None = typer.Option(
        None,
        "--gamma",
        min=0.0001,
        help="Optional obstacle heatmap near-score gamma override",
    ),
) -> None:
    """Generate obstacle heatmap predictions for a directory of images."""
    logging_ready = False
    artifact_writer: ArtifactWriter | None = None

    try:
        config = load_config(config_path)
        config = _apply_evaluation_overrides(
            config=config,
            dataset_root=dataset_root,
            images_dir=images_dir,
            predictions_dir=predictions_dir,
            output_dir=None,
        )
        config = _apply_run_video_overrides(
            config=config,
            input_path=None,
            output_dir=output_dir,
            max_frames=None,
            sample_every_n=None,
            depth_model=depth_model,
            device=device,
            fp16=fp16,
            suppression_strength=suppression_strength,
            bottom_strip_ratio=bottom_strip_ratio,
            gamma=gamma,
        )
        original_roi_enabled = config.preprocessing.roi.enabled
        config = _apply_image_prediction_overrides(
            config=config,
            keep_preprocessing_roi=keep_preprocessing_roi,
        )

        setup_logging(config.runtime.log_level)
        logging_ready = True

        resolved_images_dir = _resolve_dataset_path(
            config.evaluation.dataset.root_dir,
            config.evaluation.dataset.images_dir,
        )
        resolved_predictions_dir = _resolve_dataset_path(
            config.evaluation.dataset.root_dir,
            config.evaluation.dataset.predictions_dir,
        )
        logger.info("Starting obstacle heatmap prediction generation")
        logger.info("Input images: {}", resolved_images_dir)
        logger.info("Output predictions: {}", resolved_predictions_dir)
        logger.info("Artifact output: {}", config.output.output_dir)
        if original_roi_enabled and not config.preprocessing.roi.enabled:
            logger.info("Preprocessing ROI disabled for image batch prediction to preserve full-frame dataset context")
        logger.info("Preprocessing ROI enabled: {}", config.preprocessing.roi.enabled)
        logger.info(
            "Depth config: enabled={} provider={} model={} requested_device={} fp16={}",
            config.depth.enabled,
            config.depth.provider,
            config.depth.model,
            config.depth.device,
            config.depth.use_fp16,
        )
        logger.info(
            "Obstacle heatmap config: enabled={} mode={} suppression_strength={} bottom_strip_ratio={} gamma={}",
            config.obstacle_heatmap.enabled,
            config.obstacle_heatmap.road_suppression.mode,
            config.obstacle_heatmap.road_suppression.suppression_strength,
            config.obstacle_heatmap.road_suppression.bottom_strip_ratio,
            config.obstacle_heatmap.near_score.gamma,
        )

        image_reader = ImageDirectoryReader(
            input_dir=resolved_images_dir,
            extension=config.evaluation.dataset.file_extension_images,
        )
        preprocessor = FramePreprocessor(config.preprocessing)
        depth_estimator = create_depth_estimator(config.depth)
        obstacle_builder = create_obstacle_heatmap_builder(config.obstacle_heatmap)
        pipeline = MVPSceneAnalysisPipeline(
            preprocessor=preprocessor,
            depth_estimator=depth_estimator,
            obstacle_heatmap_builder=obstacle_builder,
        )
        artifact_writer = ArtifactWriter(config.output, config.depth, config.obstacle_heatmap)
        prediction_writer = HeatmapPredictionWriter(resolved_predictions_dir)
        runner = ImagePredictionRunner(
            pipeline=pipeline,
            image_reader=image_reader,
            prediction_writer=prediction_writer,
            artifact_writer=artifact_writer,
        )

        processed_samples = runner.run(max_images=max_images)
        logger.info(
            "Generated {} prediction file(s) in {}",
            processed_samples,
            resolved_predictions_dir,
        )
    except Exception as error:
        message = f"Prediction generation failed: {error}"
        if logging_ready:
            logger.exception(message)
        else:
            typer.echo(message, err=True)
        raise typer.Exit(code=1) from error
    finally:
        if artifact_writer is not None:
            artifact_writer.close()
