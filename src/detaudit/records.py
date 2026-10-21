"""Portable JSON artifact format for raw predictions and ground truth."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def as_list(values: Any, dtype=float) -> list:
    return np.asarray(values).astype(dtype).tolist()


def sample_record(sample: Any, detection: Any, dataset: str) -> dict[str, Any]:
    width, height = sample.image.size
    return {
        "image_id": int(sample.image_id), "dataset": dataset,
        "width": int(width), "height": int(height),
        "gt_boxes_xyxy": as_list(sample.gt_boxes, float),
        "gt_labels": as_list(sample.gt_labels, int),
        "pred_boxes_xyxy": as_list(detection.boxes, float),
        "pred_scores": as_list(detection.scores, float),
        "pred_labels": as_list(detection.labels, int),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, records: Iterable[dict[str, Any]]) -> None:
    image_ids = [int(record["image_id"]) for record in records]
    write_json(path, {"selection": "course-harness deterministic prefix", "image_ids": image_ids})
