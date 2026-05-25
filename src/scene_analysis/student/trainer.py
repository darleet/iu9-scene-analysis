from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, default_collate

from scene_analysis.evaluation.metrics import compute_precision_recall_curve_data
from scene_analysis.evaluation.visualization import plot_precision_recall_curve
from scene_analysis.student.artifacts import save_csv, save_json, save_yaml
from scene_analysis.student.config import StudentTrainConfig
from scene_analysis.student.dataset import StudentHeatmapDataset, build_resized_student_cache
from scene_analysis.student.losses import StudentHeatmapLoss
from scene_analysis.student.metrics import (
    collect_scores_and_labels,
    compute_global_average_precision,
    compute_heatmap_stats,
)
from scene_analysis.student.model import count_parameters
from scene_analysis.student.model_registry import STUDENT_REGISTRY, create_student_model, validate_student_name
from scene_analysis.student.visualization import render_training_preview
from scene_analysis.utils import safe_mkdir, to_serializable


class StudentTrainer:
    def __init__(self, config: StudentTrainConfig, student_name: str) -> None:
        self.config = config
        self.student_name = validate_student_name(student_name)
        self.model_config = config.models.get(self.student_name)
        self.output_dir = safe_mkdir(config.outputs.root_dir / config.experiment.name / self.student_name)
        self.checkpoint_dir = safe_mkdir(self.output_dir / "checkpoints")
        self.preview_dir = safe_mkdir(self.output_dir / "previews")
        self.device = self._resolve_device(config.training.device)
        self.history: list[dict[str, Any]] = []
        self.best_epoch: int | None = None
        self.best_metric: float | None = None
        self.best_val_ap: float = float("nan")
        self.parameter_count: int = 0
        self.model: torch.nn.Module | None = None
        self._last_val_scores = np.empty(0, dtype=np.float32)
        self._last_val_labels = np.empty(0, dtype=np.uint8)
        self._visual_preview_indices: list[int] | None = None

    def train(self) -> dict[str, Any]:
        self._set_seed(self.config.experiment.seed)
        save_yaml(self.output_dir / "config_resolved.yaml", self.config.model_dump(mode="json"))

        train_loader, val_loader = self._create_dataloaders()
        self.model = create_student_model(self.student_name, self.model_config).to(self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            self.model = self.model.to(memory_format=torch.channels_last)
        self.parameter_count = count_parameters(self.model)
        criterion = StudentHeatmapLoss(self.config.loss)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(self.config.training.epochs, 1),
            eta_min=self.config.scheduler.min_lr,
        )
        scaler = GradScaler("cuda", enabled=self._amp_enabled)

        logger.info(
            "Training {} ({}) params={} device={} train_samples={} val_samples={}",
            self.student_name,
            self.model_config.backbone,
            self.parameter_count,
            self.device,
            len(train_loader.dataset),
            len(val_loader.dataset),
        )

        last_train_metrics: dict[str, float] = {}
        last_val_metrics: dict[str, float] = {}
        for epoch in range(1, self.config.training.epochs + 1):
            last_train_metrics = self.train_one_epoch(
                epoch=epoch,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
            )
            last_val_metrics = self.validate(epoch=epoch, dataloader=val_loader, criterion=criterion)
            scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": last_train_metrics["train_loss"],
                "val_loss": last_val_metrics["val_loss"],
                "val_ap": last_val_metrics["val_ap"],
                "loss_bce": last_val_metrics["loss_bce"],
                "loss_dice": last_val_metrics["loss_dice"],
                "loss_distill": last_val_metrics["loss_distill"],
                "loss_offroad": last_val_metrics["loss_offroad"],
                "heatmap_mean": last_val_metrics["heatmap_mean"],
                "ignore_mean": last_val_metrics["ignore_mean"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            self.history.append(row)
            self.save_history()

            is_best = self._is_better(last_val_metrics)
            if is_best:
                self.best_epoch = epoch
                self.best_metric = self._selection_metric(last_val_metrics)
                self.best_val_ap = last_val_metrics["val_ap"]
                self.save_checkpoint(epoch, last_val_metrics, name="best.pt")

            self.save_checkpoint(epoch, last_val_metrics, name="last.pt")
            if epoch % self.config.training.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, last_val_metrics, name=f"epoch_{epoch:03d}.pt")

            if self._should_save_visual_previews(epoch):
                self.save_visual_previews(epoch, val_loader)

            logger.info(
                "Epoch {}/{}: train_loss={:.4f} val_loss={:.4f} val_ap={}",
                epoch,
                self.config.training.epochs,
                last_train_metrics["train_loss"],
                last_val_metrics["val_loss"],
                _format_float(last_val_metrics["val_ap"]),
            )

        self._save_pr_curve()
        return self.save_summary(
            train_samples=len(train_loader.dataset),
            val_samples=len(val_loader.dataset),
            last_train_metrics=last_train_metrics,
            last_val_metrics=last_val_metrics,
        )

    def train_one_epoch(
        self,
        *,
        epoch: int,
        dataloader: DataLoader[dict[str, Any]],
        criterion: StudentHeatmapLoss,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
    ) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        self.model.train()
        total_loss = 0.0
        step_count = 0
        skipped_empty_batches = 0
        skipped_sample_ids: list[str] = []
        max_batches = self.config.training.max_train_batches

        for step, batch in enumerate(dataloader, start=1):
            if max_batches is not None and step > max_batches:
                break
            if self._is_empty_valid_batch(batch):
                skipped_empty_batches += 1
                if len(skipped_sample_ids) < 5:
                    skipped_sample_ids.extend(self._sample_ids_from_batch(batch)[: 5 - len(skipped_sample_ids)])
                continue
            batch = self._move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=self._amp_enabled):
                outputs = self.model(batch["image"])
                loss, _ = criterion(
                    outputs,
                    batch["obstacle_target"],
                    batch["valid_mask"],
                    batch["ignore_mask"],
                    batch["teacher_heatmap"],
                )
            scaler.scale(loss).backward()
            if self.config.training.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu().item())
            step_count += 1
            if step % self.config.training.log_every_n_steps == 0:
                logger.info("epoch={} step={} train_loss={:.4f}", epoch, step, total_loss / step_count)

        if skipped_empty_batches:
            logger.warning(
                "Skipped {} train batch(es) with zero valid pixels at epoch {}; samples={}",
                skipped_empty_batches,
                epoch,
                skipped_sample_ids,
            )
        if step_count == 0:
            raise ValueError("Epoch contains zero train batches with valid pixels; cannot train student model")
        return {"train_loss": total_loss / max(step_count, 1)}

    def validate(
        self,
        *,
        epoch: int,
        dataloader: DataLoader[dict[str, Any]],
        criterion: StudentHeatmapLoss,
    ) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        self.model.eval()
        loss_totals = {
            "val_loss": 0.0,
            "loss_bce": 0.0,
            "loss_dice": 0.0,
            "loss_distill": 0.0,
            "loss_offroad": 0.0,
        }
        stats_totals: dict[str, float] = {
            "heatmap_min": 0.0,
            "heatmap_max": 0.0,
            "heatmap_mean": 0.0,
            "valid_mean": 0.0,
            "ignore_mean": 0.0,
            "positive_pixels": 0.0,
            "negative_pixels": 0.0,
            "valid_pixels": 0.0,
        }
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        step_count = 0
        skipped_empty_batches = 0
        skipped_sample_ids: list[str] = []
        max_batches = self.config.training.max_val_batches

        with torch.no_grad():
            for step, batch in enumerate(dataloader, start=1):
                if max_batches is not None and step > max_batches:
                    break
                if self._is_empty_valid_batch(batch):
                    skipped_empty_batches += 1
                    if len(skipped_sample_ids) < 5:
                        skipped_sample_ids.extend(self._sample_ids_from_batch(batch)[: 5 - len(skipped_sample_ids)])
                    continue
                batch = self._move_batch(batch)
                outputs = self.model(batch["image"])
                loss, parts = criterion(
                    outputs,
                    batch["obstacle_target"],
                    batch["valid_mask"],
                    batch["ignore_mask"],
                    batch["teacher_heatmap"],
                )
                loss_totals["val_loss"] += float(loss.detach().cpu().item())
                for name, value in parts.items():
                    loss_totals[name] += float(value.detach().cpu().item())
                stats = compute_heatmap_stats(
                    outputs["final_heatmap"],
                    batch["valid_mask"],
                    batch["ignore_mask"],
                    batch["obstacle_target"],
                )
                for key in stats_totals:
                    stats_totals[key] += float(stats[key])
                if self.config.validation.compute_average_precision:
                    scores, labels = collect_scores_and_labels(
                        outputs["final_heatmap"],
                        batch["obstacle_target"],
                        batch["valid_mask"],
                    )
                    all_scores.append(scores)
                    all_labels.append(labels)
                step_count += 1

        if skipped_empty_batches:
            logger.warning(
                "Skipped {} validation batch(es) with zero valid pixels at epoch {}; samples={}",
                skipped_empty_batches,
                epoch,
                skipped_sample_ids,
            )
        if step_count == 0:
            raise ValueError("Validation contains zero batches with valid pixels; cannot evaluate student model")

        averaged = {key: value / max(step_count, 1) for key, value in loss_totals.items()}
        for key in ("heatmap_min", "heatmap_max", "heatmap_mean", "valid_mean", "ignore_mean"):
            averaged[key] = stats_totals[key] / max(step_count, 1)
        averaged["positive_pixels"] = stats_totals["positive_pixels"]
        averaged["negative_pixels"] = stats_totals["negative_pixels"]
        averaged["valid_pixels"] = stats_totals["valid_pixels"]

        if all_scores and all_labels:
            self._last_val_scores = np.concatenate(all_scores)
            self._last_val_labels = np.concatenate(all_labels)
            averaged["val_ap"] = compute_global_average_precision(self._last_val_scores, self._last_val_labels)
            if math.isnan(averaged["val_ap"]):
                logger.warning("Validation AP is undefined at epoch {}: need both positive and negative pixels", epoch)
        else:
            self._last_val_scores = np.empty(0, dtype=np.float32)
            self._last_val_labels = np.empty(0, dtype=np.uint8)
            averaged["val_ap"] = float("nan")
        return averaged

    def save_checkpoint(self, epoch: int, metrics: dict[str, float], *, name: str) -> Path:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        path = self.checkpoint_dir / name
        if not self.config.outputs.save_checkpoints:
            return path
        checkpoint = {
            "student_name": self.student_name,
            "backbone": self.model_config.backbone,
            "model_state_dict": self.model.state_dict(),
            "input_height": self.config.input.height,
            "input_width": self.config.input.width,
            "normalize_mean": self.config.input.normalize_mean,
            "normalize_std": self.config.input.normalize_std,
            "epoch": epoch,
            "val_ap": metrics.get("val_ap", float("nan")),
            "parameter_count": self.parameter_count,
            "config": self.config.model_dump(mode="json"),
        }
        torch.save(checkpoint, path)
        return path

    def save_history(self) -> Path:
        path = self.output_dir / "history.csv"
        if not self.config.outputs.save_history_csv:
            return path
        return save_csv(
            path,
            self.history,
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_ap",
                "loss_bce",
                "loss_dice",
                "loss_distill",
                "loss_offroad",
                "heatmap_mean",
                "ignore_mean",
                "lr",
            ],
        )

    def save_summary(
        self,
        *,
        train_samples: int,
        val_samples: int,
        last_train_metrics: dict[str, float],
        last_val_metrics: dict[str, float],
    ) -> dict[str, Any]:
        summary = {
            "experiment_name": self.config.experiment.name,
            "student_name": self.student_name,
            "backbone": self.model_config.backbone,
            "parameter_count": self.parameter_count,
            "epochs": self.config.training.epochs,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "best_epoch": self.best_epoch,
            "best_val_ap": self.best_val_ap,
            "last_train_loss": last_train_metrics.get("train_loss"),
            "last_val_loss": last_val_metrics.get("val_loss"),
            "heatmap_min": last_val_metrics.get("heatmap_min"),
            "heatmap_max": last_val_metrics.get("heatmap_max"),
            "heatmap_mean": last_val_metrics.get("heatmap_mean"),
            "ignore_mean": last_val_metrics.get("ignore_mean"),
            "checkpoint_best": str(self.checkpoint_dir / "best.pt"),
            "checkpoint_last": str(self.checkpoint_dir / "last.pt"),
            "summary": str(self.output_dir / "summary.json"),
            "history": str(self.output_dir / "history.csv"),
            "status": "ok",
        }
        if self.config.outputs.save_summary_json:
            save_json(self.output_dir / "summary.json", summary)
        return to_serializable(summary)

    def save_visual_previews(self, epoch: int, dataloader: DataLoader[dict[str, Any]]) -> Path | None:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        batch = self._sample_visual_preview_batch(epoch, dataloader)
        batch_on_device = self._move_batch(batch)
        with torch.no_grad():
            outputs = self.model(batch_on_device["image"])
        output_path = self.preview_dir / f"epoch_{epoch:03d}_sample_grid.png"
        return render_training_preview(
            batch_on_device,
            outputs,
            output_path,
            self.config.input.normalize_mean,
            self.config.input.normalize_std,
            max_samples=min(self.config.validation.num_visual_examples, 4),
        )

    def _sample_visual_preview_batch(
        self,
        epoch: int,
        dataloader: DataLoader[dict[str, Any]],
    ) -> dict[str, Any]:
        dataset = getattr(dataloader, "dataset", None)
        max_samples = min(self.config.validation.num_visual_examples, 4)
        if dataset is None or max_samples <= 0:
            return next(iter(dataloader))
        dataset_length = len(dataset)
        if dataset_length <= 0:
            return next(iter(dataloader))

        sample_count = min(max_samples, dataset_length)
        if self._visual_preview_indices is None:
            rng = random.Random(self.config.experiment.seed)
            self._visual_preview_indices = rng.sample(range(dataset_length), sample_count)
        indices = self._visual_preview_indices[:sample_count]
        return default_collate([dataset[index] for index in indices])

    def _create_dataloaders(self) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
        self._build_resized_caches()
        train_dataset = StudentHeatmapDataset(
            self.config.dataset.prepared_root_dir,
            "train",
            self.config.dataset,
            self.config.input,
            self.config.augmentations,
            training=True,
        )
        val_dataset = StudentHeatmapDataset(
            self.config.dataset.prepared_root_dir,
            "val",
            self.config.dataset,
            self.config.input,
            self.config.augmentations,
            training=False,
        )
        generator = torch.Generator()
        generator.manual_seed(self.config.experiment.seed)
        loader_options: dict[str, Any] = {
            "batch_size": self.config.training.batch_size,
            "num_workers": self.config.training.num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if self.config.training.num_workers > 0:
            loader_options["persistent_workers"] = True
            loader_options["prefetch_factor"] = 4
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            generator=generator,
            **loader_options,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_options,
        )
        return train_loader, val_loader

    def _build_resized_caches(self) -> None:
        if not self.config.dataset.use_resized_cache:
            return
        logger.info(
            "Preparing resized student cache at {}x{}",
            self.config.input.height,
            self.config.input.width,
        )
        for split in ("train", "val"):
            summary = build_resized_student_cache(
                self.config.dataset.prepared_root_dir,
                split,
                self.config.dataset,
                self.config.input,
            )
            logger.info(
                "Resized cache split={} total={} created={} skipped={} dir={}",
                summary["split"],
                summary["total"],
                summary["created"],
                summary["skipped"],
                summary["cache_dir"],
            )

    def _should_save_visual_previews(self, epoch: int) -> bool:
        if not self.config.validation.save_visual_examples or not self.config.outputs.save_visual_previews:
            return False
        return epoch == 1 or epoch % self.config.validation.save_visual_every_n_epochs == 0

    @staticmethod
    def _is_empty_valid_batch(batch: dict[str, Any]) -> bool:
        valid_mask = batch.get("valid_mask")
        if not isinstance(valid_mask, torch.Tensor):
            return False
        return float(valid_mask.sum().item()) <= 0.0

    @staticmethod
    def _sample_ids_from_batch(batch: dict[str, Any]) -> list[str]:
        sample_ids = batch.get("sample_id")
        if sample_ids is None:
            return []
        if isinstance(sample_ids, str):
            return [sample_ids]
        if isinstance(sample_ids, (list, tuple)):
            return [str(item) for item in sample_ids]
        return [str(sample_ids)]

    def _is_better(self, metrics: dict[str, float]) -> bool:
        metric = self._selection_metric(metrics)
        if self.best_metric is None:
            return True
        if self.config.validation.save_best_by == "val_loss":
            return metric < self.best_metric
        if math.isnan(metric):
            return False
        if math.isnan(self.best_metric):
            return True
        return metric > self.best_metric

    def _selection_metric(self, metrics: dict[str, float]) -> float:
        if self.config.validation.save_best_by == "val_loss":
            return float(metrics["val_loss"])
        return float(metrics["val_ap"])

    def _save_pr_curve(self) -> Path:
        path = self.output_dir / "pr_curve.png"
        if not self.config.outputs.save_pr_curve_png:
            return path
        precision, recall, _ = compute_precision_recall_curve_data(self._last_val_scores, self._last_val_labels)
        plot_precision_recall_curve(precision, recall, self.best_val_ap, path)
        return path

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                moved[key] = value
                continue
            if key == "image" and self.device.type == "cuda":
                moved[key] = value.to(
                    self.device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            else:
                moved[key] = value.to(self.device, non_blocking=True)
        return moved

    @property
    def _amp_enabled(self) -> bool:
        return bool(self.config.training.use_amp and self.device.type == "cuda")

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        normalized = requested.strip().lower()
        if normalized == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                logger.warning("CUDA is not available; using MPS. Full training is still recommended on CUDA GPU.")
                return torch.device("mps")
            logger.warning("CUDA is not available; CPU training will be slow and is intended for smoke/debug runs.")
            return torch.device("cpu")
        device = torch.device(normalized)
        if device.type != "cuda":
            logger.warning("CUDA is not selected; training will be slow and is intended for smoke/debug runs.")
        return device


def _format_float(value: float) -> str:
    return "n/a" if math.isnan(float(value)) else f"{float(value):.4f}"
