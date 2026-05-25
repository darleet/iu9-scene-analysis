from __future__ import annotations

import argparse
import gc
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from loguru import logger

from scene_analysis.logging_setup import setup_logging
from scene_analysis.student.artifacts import save_csv, save_json, save_yaml
from scene_analysis.student.config import StudentTrainConfig, load_student_train_config
from scene_analysis.student.model_registry import validate_student_name
from scene_analysis.student.trainer import StudentTrainer


BASE_VALUES: dict[str, float | int] = {
    "loss.bce_weight": 1.0,
    "loss.dice_weight": 1.0,
    "loss.roi_bce_weight": 0.15,
    "loss.distill_mse_weight": 0.02,
    "loss.offroad_weight": 0.02,
    "loss.positive_class_weight": 6.0,
    "training.batch_size": 4,
}

SEARCH_SPACE: list[tuple[str, list[float | int]]] = [
    ("loss.bce_weight", [0.5, 1.0, 1.5]),
    ("loss.dice_weight", [0.5, 1.0, 1.5]),
    ("loss.roi_bce_weight", [0.0, 0.15, 0.30]),
    ("loss.distill_mse_weight", [0.0, 0.02, 0.05]),
    ("loss.offroad_weight", [0.0, 0.02, 0.05]),
    ("loss.positive_class_weight", [3.0, 6.0, 12.0]),
    ("training.batch_size", [2, 4, 8]),
]

DEFAULT_STUDENT = "student_s"


@dataclass(frozen=True)
class TrialResult:
    stage: int
    parameter: str
    candidate_value: float | int
    experiment_name: str
    artifact_dir: Path
    score: float
    status: str
    students: list[dict[str, Any]]
    error: str | None = None


def main() -> None:
    args = _parse_args()
    setup_logging(args.log_level)

    config_path = args.config.resolve()
    base_config = load_student_train_config(config_path)
    student = validate_student_name(args.student)
    tuning_name = args.name or f"student_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tuning_root = (args.output_root or base_config.outputs.root_dir / tuning_name).expanduser()

    current_values = dict(BASE_VALUES)
    all_trials: list[TrialResult] = []
    stage_summaries: list[dict[str, Any]] = []

    logger.info("Tuning run: {}", tuning_name)
    logger.info("Artifacts root: {}", tuning_root)
    logger.info("Student per trial: {}", student)
    logger.info("Base config: {}", config_path)

    _write_plan(tuning_root, tuning_name, config_path, student, args, current_values)

    if args.dry_run:
        logger.info("Dry run only. No training will be started.")
        return

    for stage_index, (parameter, candidates) in enumerate(SEARCH_SPACE, start=1):
        stage_trials: list[TrialResult] = []
        fixed_before = dict(current_values)
        logger.info(
            "Stage {}/{}: tuning {} with candidates={}",
            stage_index,
            len(SEARCH_SPACE),
            parameter,
            candidates,
        )

        for candidate_index, candidate_value in enumerate(candidates, start=1):
            trial_config = _make_trial_config(
                base_config=base_config,
                tuning_root=tuning_root,
                tuning_name=tuning_name,
                stage_index=stage_index,
                candidate_index=candidate_index,
                parameter=parameter,
                candidate_value=candidate_value,
                current_values=current_values,
                args=args,
            )
            experiment_name = trial_config.experiment.name
            artifact_dir = trial_config.outputs.root_dir / experiment_name
            save_yaml(tuning_root / "trial_configs" / f"{experiment_name}.yaml", trial_config.model_dump(mode="json"))

            result = _run_trial(
                config=trial_config,
                student=student,
                stage=stage_index,
                parameter=parameter,
                candidate_value=candidate_value,
                experiment_name=experiment_name,
                artifact_dir=artifact_dir,
            )
            stage_trials.append(result)
            all_trials.append(result)
            _write_progress(tuning_root, all_trials, stage_summaries, current_values)

        selected = _select_best_trial(stage_trials)
        if selected is None:
            raise RuntimeError(f"All trials failed at stage {stage_index} for parameter {parameter}")

        current_values[parameter] = selected.candidate_value
        stage_summary = {
            "stage": stage_index,
            "parameter": parameter,
            "fixed_values_before": fixed_before,
            "selected_value": selected.candidate_value,
            "selected_score": selected.score,
            "selected_experiment_name": selected.experiment_name,
            "trials": [_trial_to_dict(trial) for trial in stage_trials],
        }
        stage_summaries.append(stage_summary)
        _write_progress(tuning_root, all_trials, stage_summaries, current_values)
        logger.info(
            "Selected {}={} at stage {} with mean best_val_ap={:.6f}",
            parameter,
            selected.candidate_value,
            stage_index,
            selected.score,
        )

    final_config = _config_with_values(base_config.model_copy(deep=True), current_values)
    final_config.outputs.root_dir = tuning_root
    final_config.experiment.name = f"{tuning_name}_final_config"
    if args.epochs is not None:
        final_config.training.epochs = args.epochs
    if args.device is not None:
        final_config.training.device = args.device
    final_config.models.train_students = [student]
    save_yaml(tuning_root / "best_config.yaml", final_config.model_dump(mode="json"))
    if args.train_all_final:
        final_run_config = final_config.model_copy(deep=True)
        final_run_config.experiment.name = f"{tuning_name}_final_all_students"
        final_run_config.models.train_students = ["student_s", "student_m", "student_q"]
        save_yaml(tuning_root / "final_all_students_config.yaml", final_run_config.model_dump(mode="json"))
        for student_name in final_run_config.models.train_students:
            trainer = StudentTrainer(final_run_config.model_copy(deep=True), student_name)
            trainer.train()
            _cleanup_cuda()
    _write_progress(tuning_root, all_trials, stage_summaries, current_values)
    logger.info("Tuning complete. Best config: {}", tuning_root / "best_config.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Coordinate-search tuning for one student model. Each candidate trains one selected student, "
            "then the best candidate is selected by best_val_ap."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/student_train.yaml"))
    parser.add_argument("--name", type=str, default=None, help="Tuning run name. Defaults to timestamped name.")
    parser.add_argument("--output-root", type=Path, default=None, help="Root folder for all tuning artifacts.")
    parser.add_argument("--student", type=str, default=DEFAULT_STUDENT, help="Student to tune.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count for every trial.")
    parser.add_argument("--device", type=str, default=None, help="Override training device, e.g. cuda or cpu.")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment seed for every trial.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Debug limit for train batches.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Debug limit for validation batches.")
    parser.add_argument(
        "--train-all-final",
        action="store_true",
        help="After tuning, train student_s, student_m and student_q once with the selected values.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the plan and exit without training.")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _write_plan(
    tuning_root: Path,
    tuning_name: str,
    config_path: Path,
    student: str,
    args: argparse.Namespace,
    current_values: dict[str, float | int],
) -> None:
    save_json(
        tuning_root / "tuning_plan.json",
        {
            "tuning_name": tuning_name,
            "config": str(config_path),
            "student": student,
            "base_values": current_values,
            "search_space": [{"parameter": name, "candidates": values} for name, values in SEARCH_SPACE],
            "epochs_override": args.epochs,
            "device_override": args.device,
            "seed_override": args.seed,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "train_all_final": args.train_all_final,
            "dry_run": args.dry_run,
        },
    )


def _make_trial_config(
    *,
    base_config: StudentTrainConfig,
    tuning_root: Path,
    tuning_name: str,
    stage_index: int,
    candidate_index: int,
    parameter: str,
    candidate_value: float | int,
    current_values: dict[str, float | int],
    args: argparse.Namespace,
) -> StudentTrainConfig:
    trial_values = dict(current_values)
    trial_values[parameter] = candidate_value
    config = _config_with_values(base_config.model_copy(deep=True), trial_values)
    config.outputs.root_dir = tuning_root
    config.experiment.name = _trial_name(tuning_name, stage_index, candidate_index, parameter, candidate_value)
    config.models.train_students = [validate_student_name(args.student)]
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.device is not None:
        config.training.device = args.device
    if args.seed is not None:
        config.experiment.seed = args.seed
    if args.max_train_batches is not None:
        config.training.max_train_batches = args.max_train_batches
    if args.max_val_batches is not None:
        config.training.max_val_batches = args.max_val_batches
    return config


def _config_with_values(config: StudentTrainConfig, values: dict[str, float | int]) -> StudentTrainConfig:
    for dotted_path, value in values.items():
        _set_dotted_value(config, dotted_path, value)
    return config


def _set_dotted_value(config: StudentTrainConfig, dotted_path: str, value: float | int) -> None:
    target: Any = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _run_trial(
    *,
    config: StudentTrainConfig,
    student: str,
    stage: int,
    parameter: str,
    candidate_value: float | int,
    experiment_name: str,
    artifact_dir: Path,
) -> TrialResult:
    student_results: list[dict[str, Any]] = []
    error: str | None = None
    logger.info("Starting trial {}: {}={}", experiment_name, parameter, candidate_value)
    try:
        trainer = StudentTrainer(config.model_copy(deep=True), student)
        summary = trainer.train()
        student_results.append(summary)
        _cleanup_cuda()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("Trial {} failed: {}", experiment_name, exc)
        _cleanup_cuda()

    score = _trial_score(student_results, error=error)
    status = "ok" if error is None and math.isfinite(score) else "failed"
    return TrialResult(
        stage=stage,
        parameter=parameter,
        candidate_value=candidate_value,
        experiment_name=experiment_name,
        artifact_dir=artifact_dir,
        score=score,
        status=status,
        students=student_results,
        error=error,
    )


def _trial_score(student_results: list[dict[str, Any]], *, error: str | None) -> float:
    if error is not None or len(student_results) != 1:
        return float("nan")
    score = float(student_results[0].get("best_val_ap", float("nan")))
    if not math.isfinite(score):
        return float("nan")
    return score


def _select_best_trial(trials: list[TrialResult]) -> TrialResult | None:
    valid_trials = [trial for trial in trials if trial.status == "ok" and math.isfinite(trial.score)]
    if not valid_trials:
        return None
    return max(valid_trials, key=lambda trial: trial.score)


def _write_progress(
    tuning_root: Path,
    trials: list[TrialResult],
    stage_summaries: list[dict[str, Any]],
    current_values: dict[str, float | int],
) -> None:
    save_json(
        tuning_root / "tuning_summary.json",
        {
            "current_best_values": current_values,
            "stages": stage_summaries,
            "trials": [_trial_to_dict(trial) for trial in trials],
        },
    )
    save_csv(
        tuning_root / "trials.csv",
        [_trial_csv_row(trial) for trial in trials],
        [
            "stage",
            "parameter",
            "candidate_value",
            "experiment_name",
            "score",
            "status",
            "artifact_dir",
            "error",
        ],
    )


def _trial_to_dict(trial: TrialResult) -> dict[str, Any]:
    return {
        "stage": trial.stage,
        "parameter": trial.parameter,
        "candidate_value": trial.candidate_value,
        "experiment_name": trial.experiment_name,
        "artifact_dir": str(trial.artifact_dir),
        "score": trial.score,
        "status": trial.status,
        "students": trial.students,
        "error": trial.error,
    }


def _trial_csv_row(trial: TrialResult) -> dict[str, Any]:
    return {
        "stage": trial.stage,
        "parameter": trial.parameter,
        "candidate_value": trial.candidate_value,
        "experiment_name": trial.experiment_name,
        "score": trial.score,
        "status": trial.status,
        "artifact_dir": str(trial.artifact_dir),
        "error": trial.error or "",
    }


def _trial_name(
    tuning_name: str,
    stage_index: int,
    candidate_index: int,
    parameter: str,
    value: float | int,
) -> str:
    short_parameter = parameter.replace(".", "_")
    return f"{tuning_name}_s{stage_index:02d}_c{candidate_index}_{short_parameter}_{_format_value(value)}"


def _format_value(value: float | int) -> str:
    text = str(value)
    return text.replace("-", "m").replace(".", "p")


def _cleanup_cuda() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
