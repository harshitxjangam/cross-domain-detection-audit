"""Measure the recall mechanism of an NMS sweep from saved predictions.

This is intentionally a stratum-conditioned recall analysis, not a second AP
benchmark. It asks whether relaxed final NMS preserves detections specifically
for same-class overlapping ground-truth objects. Predictions are score-ordered,
limited to the saved metric cap, and matched one-to-one at the requested IoU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze import iou_matrix
from .records import write_json


STRATA = ("isolated", "near", "overlap")


def same_class_density_bins(record: dict) -> np.ndarray:
    """Return the same-class neighborhood bin for every ground-truth box."""
    boxes = np.asarray(record["gt_boxes_xyxy"], dtype=float).reshape(-1, 4)
    labels = np.asarray(record["gt_labels"], dtype=int).reshape(-1)
    overlaps = iou_matrix(boxes)
    result = []
    for index, label in enumerate(labels):
        same = labels == label
        same[index] = False
        max_iou = float(overlaps[index, same].max()) if same.any() else 0.0
        if max_iou >= 0.30:
            result.append("overlap")
        elif max_iou >= 0.10:
            result.append("near")
        else:
            result.append("isolated")
    return np.asarray(result, dtype=object)


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    left = np.maximum(box[:2], boxes[:, :2])
    right = np.minimum(box[2:], boxes[:, 2:])
    wh = np.clip(right - left, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    area_one = np.prod(np.clip(box[2:] - box[:2], 0, None))
    area_many = np.prod(np.clip(boxes[:, 2:] - boxes[:, :2], 0, None), axis=1)
    return inter / np.maximum(area_one + area_many - inter, 1e-9)


def same_class_stratum_recall(records: list[dict], *, iou_threshold: float = 0.50,
                              max_detections: int = 100) -> dict[str, dict[str, float | int]]:
    """Compute score-greedy, one-to-one recall by same-class density stratum."""
    totals = {name: 0 for name in STRATA}
    matches = {name: 0 for name in STRATA}
    for record in records:
        gt_boxes = np.asarray(record["gt_boxes_xyxy"], dtype=float).reshape(-1, 4)
        gt_labels = np.asarray(record["gt_labels"], dtype=int).reshape(-1)
        pred_boxes = np.asarray(record["pred_boxes_xyxy"], dtype=float).reshape(-1, 4)
        pred_scores = np.asarray(record["pred_scores"], dtype=float).reshape(-1)
        pred_labels = np.asarray(record["pred_labels"], dtype=int).reshape(-1)
        density = same_class_density_bins(record)
        order = np.argsort(-pred_scores, kind="mergesort")[:max_detections]
        pred_boxes, pred_labels = pred_boxes[order], pred_labels[order]
        assigned: set[int] = set()
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            candidates = [index for index, label in enumerate(gt_labels)
                          if label == pred_label and index not in assigned]
            if not candidates:
                continue
            ious = _iou_one_to_many(pred_box, gt_boxes[candidates])
            best = int(np.argmax(ious))
            if ious[best] >= iou_threshold:
                assigned.add(candidates[best])
        for index, name in enumerate(density):
            totals[str(name)] += 1
            if index in assigned:
                matches[str(name)] += 1
    return {
        name: {"matched": matches[name], "total": totals[name],
               "recall": matches[name] / totals[name] if totals[name] else 0.0}
        for name in STRATA
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute NMS-sweep recall by same-class density stratum.")
    parser.add_argument("records", type=Path, help="Saved records.json from one completed run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-detections", type=int, default=100)
    args = parser.parse_args()
    records = json.loads(args.records.read_text(encoding="utf-8"))
    result = {
        "source_records": str(args.records),
        "num_images": len(records),
        "matching": {"iou_threshold": args.iou, "max_detections": args.max_detections,
                     "policy": "score-ordered greedy one-to-one matching"},
        "same_class_stratum_recall": same_class_stratum_recall(
            records, iou_threshold=args.iou, max_detections=args.max_detections),
    }
    write_json(args.out, result)


if __name__ == "__main__":
    main()
