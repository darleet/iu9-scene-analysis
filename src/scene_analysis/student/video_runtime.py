from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from scene_analysis.obstacle_map.visualization import heatmap_to_bgr
from scene_analysis.student.artifacts import append_jsonl, make_timestamped_run_dir, save_json
from scene_analysis.student.config import StudentInferenceConfig
from scene_analysis.student.inference import StudentInferenceRunner
from scene_analysis.student.model_registry import validate_student_name
from scene_analysis.utils import safe_mkdir


def run_student_on_video_folder(
    config: StudentInferenceConfig,
    student_name: str,
    checkpoint_path: Path,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    student = validate_student_name(student_name)
    resolved_input_dir = (input_dir or config.video_folder.input_dir).expanduser()
    base_output_dir = (output_dir or config.video_folder.output_dir).expanduser()
    if not resolved_input_dir.exists() or not resolved_input_dir.is_dir():
        raise FileNotFoundError(f"Video input directory not found: {resolved_input_dir}")

    run_dir = make_timestamped_run_dir(base_output_dir, student)
    videos_dir = safe_mkdir(run_dir / "videos")
    frames_dir = safe_mkdir(run_dir / "frames")
    heatmaps_png_dir = safe_mkdir(run_dir / "heatmaps_png")
    heatmaps_npy_dir = safe_mkdir(run_dir / "heatmaps_npy")
    results_path = run_dir / "results.jsonl"

    videos = _discover_videos(resolved_input_dir, config.video_folder.video_extensions)
    if not videos:
        raise FileNotFoundError(f"No video files found in {resolved_input_dir}")

    runner = StudentInferenceRunner(config, student, checkpoint_path)
    runner.load_checkpoint(checkpoint_path)

    frames_processed = 0
    inference_times: list[float] = []
    started = time.perf_counter()
    for video_path in videos:
        video_frames, video_times = _process_video(
            config=config,
            runner=runner,
            video_path=video_path,
            videos_dir=videos_dir,
            frames_dir=frames_dir,
            heatmaps_png_dir=heatmaps_png_dir,
            heatmaps_npy_dir=heatmaps_npy_dir,
            results_path=results_path,
        )
        frames_processed += video_frames
        inference_times.extend(video_times)

    elapsed = max(time.perf_counter() - started, 1e-6)
    summary = {
        "student": student,
        "checkpoint": str(checkpoint_path),
        "videos_processed": len(videos),
        "frames_processed": frames_processed,
        "avg_fps": frames_processed / elapsed,
        "avg_inference_ms": float(np.mean(inference_times)) if inference_times else 0.0,
        "output_dir": str(run_dir),
    }
    save_json(run_dir / "summary.json", summary)
    return summary


def run_student_on_camera(
    config: StudentInferenceConfig,
    student_name: str,
    checkpoint_path: Path,
    camera_index: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    student = validate_student_name(student_name)
    index = config.camera.camera_index if camera_index is None else int(camera_index)
    base_output_dir = (output_dir or config.camera.output_dir).expanduser()
    run_dir = make_timestamped_run_dir(base_output_dir, student)
    results_path = run_dir / "results.jsonl"

    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Camera index {index} is not available. Try --camera-index 1 or check permissions.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
    capture.set(cv2.CAP_PROP_FPS, config.camera.fps)

    runner = StudentInferenceRunner(config, student, checkpoint_path)
    runner.load_checkpoint(checkpoint_path)

    writer: cv2.VideoWriter | None = None
    if config.camera.save_video:
        writer = _create_video_writer(
            run_dir / "camera_overlay.mp4",
            fps=float(config.camera.fps),
            size=(config.camera.width, config.camera.height),
        )

    frames_processed = 0
    inference_times: list[float] = []
    started = time.perf_counter()
    last_report = started
    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break
            result = runner.predict_frame(frame_bgr)
            stats = result["stats"]
            overlay = runner.render_overlay(frame_bgr, result["heatmap"], stats, frame_index=frames_processed)
            inference_times.append(float(stats["inference_ms"]))
            if writer is not None:
                writer.write(overlay)
            if config.output.save_jsonl:
                append_jsonl(
                    results_path,
                    {
                        "video": "camera",
                        "frame_index": frames_processed,
                        "timestamp_ms": frames_processed * 1000.0 / max(float(config.camera.fps), 1.0),
                        "student": student,
                        "checkpoint": str(checkpoint_path),
                        **stats,
                    },
                )
            frames_processed += 1

            now = time.perf_counter()
            if now - last_report >= 3.0:
                fps = frames_processed / max(now - started, 1e-6)
                logger.info("Live student camera FPS: {:.2f}", fps)
                last_report = now

            if config.camera.display:
                cv2.imshow("student obstacle heatmap", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if config.camera.display:
            cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - started, 1e-6)
    summary = {
        "student": student,
        "checkpoint": str(checkpoint_path),
        "camera_index": index,
        "frames_processed": frames_processed,
        "avg_fps": frames_processed / elapsed,
        "avg_inference_ms": float(np.mean(inference_times)) if inference_times else 0.0,
        "output_dir": str(run_dir),
    }
    save_json(run_dir / "summary.json", summary)
    return summary


def _process_video(
    *,
    config: StudentInferenceConfig,
    runner: StudentInferenceRunner,
    video_path: Path,
    videos_dir: Path,
    frames_dir: Path,
    heatmaps_png_dir: Path,
    heatmaps_npy_dir: Path,
    results_path: Path,
) -> tuple[int, list[float]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = fps if fps > 0 else 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer: cv2.VideoWriter | None = None
    if config.output.save_overlay_video:
        writer = _create_video_writer(videos_dir / f"{video_path.stem}_overlay.mp4", fps=fps, size=(width, height))

    frame_count = 0
    processed_count = 0
    inference_times: list[float] = []
    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break
            current_index = frame_count
            frame_count += 1
            if current_index % config.video_folder.sample_every_n != 0:
                continue
            if config.video_folder.max_frames is not None and processed_count >= config.video_folder.max_frames:
                break

            result = runner.predict_frame(frame_bgr)
            stats = result["stats"]
            heatmap = result["heatmap"]
            overlay = runner.render_overlay(frame_bgr, heatmap, stats, frame_index=current_index)
            if writer is not None:
                writer.write(overlay)
            _save_optional_video_artifacts(
                config=config,
                video_stem=video_path.stem,
                frame_index=current_index,
                overlay=overlay,
                heatmap=heatmap,
                frames_dir=frames_dir,
                heatmaps_png_dir=heatmaps_png_dir,
                heatmaps_npy_dir=heatmaps_npy_dir,
            )
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or (current_index * 1000.0 / fps))
            if config.output.save_jsonl:
                append_jsonl(
                    results_path,
                    {
                        "video": video_path.name,
                        "frame_index": current_index,
                        "timestamp_ms": timestamp_ms,
                        "student": runner.student_name,
                        "checkpoint": str(runner.checkpoint_path),
                        **stats,
                    },
                )
            inference_times.append(float(stats["inference_ms"]))
            processed_count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return processed_count, inference_times


def _save_optional_video_artifacts(
    *,
    config: StudentInferenceConfig,
    video_stem: str,
    frame_index: int,
    overlay: np.ndarray,
    heatmap: np.ndarray,
    frames_dir: Path,
    heatmaps_png_dir: Path,
    heatmaps_npy_dir: Path,
) -> None:
    stem = f"{video_stem}_frame_{frame_index:06d}"
    if config.output.save_frames:
        cv2.imwrite(str(frames_dir / f"{stem}.png"), overlay)
    if config.output.save_heatmap_png:
        cv2.imwrite(str(heatmaps_png_dir / f"{stem}.png"), heatmap_to_bgr(heatmap, config.visualization.colormap))
    if config.output.save_heatmap_npy:
        np.save(heatmaps_npy_dir / f"{stem}.npy", heatmap.astype(np.float32, copy=False))


def _discover_videos(input_dir: Path, extensions: list[str]) -> list[Path]:
    normalized_extensions = {extension.lower() for extension in extensions}
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in normalized_extensions)


def _create_video_writer(path: Path, *, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    safe_mkdir(path.parent)
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video output size for {path}: {size}")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Failed to create output video writer: {path}")
    return writer
