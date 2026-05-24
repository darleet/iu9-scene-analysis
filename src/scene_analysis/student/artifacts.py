from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from scene_analysis.utils import safe_mkdir, to_serializable


def make_timestamped_run_dir(base_dir: Path, student_name: str) -> Path:
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return safe_mkdir(base_dir.expanduser() / run_name / student_name)


def save_json(path: Path, payload: dict[str, Any]) -> Path:
    safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_serializable(payload), file, ensure_ascii=False, indent=2)
    return path


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(to_serializable(payload), ensure_ascii=False) + "\n")


def save_yaml(path: Path, payload: dict[str, Any]) -> Path:
    safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(to_serializable(payload), file, sort_keys=False, allow_unicode=True)
    return path


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(to_serializable(row))
    return path
