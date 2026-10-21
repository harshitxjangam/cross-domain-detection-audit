"""Model-independent overlap/density summaries for saved evaluation records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .records import write_json
from .taxonomy import CLASSES


def iou_matrix(boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.zeros((0, 0), dtype=np.float32)
    left = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    right = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    wh = np.clip(right - left, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area = np.prod(np.clip(boxes[:, 2:] - boxes[:, :2], 0, None), axis=1)
    return inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)


def strata(records: list[dict]) -> dict:
    counts = Counter()
    for record in records:
        boxes = np.asarray(record["gt_boxes_xyxy"], dtype=float).reshape(-1, 4)
        labels = np.asarray(record["gt_labels"], dtype=int)
        overlaps = iou_matrix(boxes)
        for index, (box, label) in enumerate(zip(boxes, labels)):
            area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
            scale = "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"
            same = (labels == label).copy(); same[index] = False
            all_other = np.ones(len(boxes), dtype=bool); all_other[index] = False
            same_max = float(overlaps[index, same].max()) if same.any() else 0.0
            all_max = float(overlaps[index, all_other].max()) if all_other.any() else 0.0
            same_bin = "same_overlap" if same_max >= .30 else "same_near" if same_max >= .10 else "same_isolated"
            all_bin = "all_overlap" if all_max >= .30 else "all_near" if all_max >= .10 else "all_isolated"
            counts[f"scale/{scale}"] += 1
            counts[f"same_class/{same_bin}"] += 1
            counts[f"all_class/{all_bin}"] += 1
            counts[f"class/{CLASSES[int(label)]}"] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GT strata from audit records.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.records.read_text(encoding="utf-8"))
    write_json(args.out, {"num_images": len(records), "gt_strata_counts": strata(records)})


if __name__ == "__main__":
    main()
