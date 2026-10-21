"""Run TIDE error decomposition on a saved evaluation ``records.json`` file.

This command never re-runs a detector.  It makes the failure analysis
reproducible from exactly the post-processing outputs used for the COCO
metrics, which is important when comparing NMS settings.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .records import sha256, write_json
from .taxonomy import CLASSES


def coco_payload(records: list[dict]) -> tuple[dict, list[dict]]:
    """Convert the portable record format to minimal COCO GT/result JSON."""
    images, annotations, detections = [], [], []
    ann_id = 1
    for record in records:
        image_id = int(record["image_id"])
        images.append({"id": image_id, "file_name": str(image_id),
                       "width": int(record["width"]), "height": int(record["height"])})
        for box, label in zip(record["gt_boxes_xyxy"], record["gt_labels"]):
            x1, y1, x2, y2 = map(float, box)
            width, height = x2 - x1, y2 - y1
            annotations.append({
                "id": ann_id, "image_id": image_id, "category_id": int(label) + 1,
                "bbox": [x1, y1, width, height], "area": width * height, "iscrowd": 0,
                # TIDE's COCO reader requires a segmentation even in BOX mode.
                # A rectangular polygon is exact for the only annotation type
                # retained by this 2D bounding-box audit.
                "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
            })
            ann_id += 1
        for box, score, label in zip(record["pred_boxes_xyxy"], record["pred_scores"], record["pred_labels"]):
            x1, y1, x2, y2 = map(float, box)
            detections.append({
                "image_id": image_id, "category_id": int(label) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score),
            })
    ground_truth = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": idx + 1, "name": name} for idx, name in enumerate(CLASSES)],
    }
    return ground_truth, detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a machine-readable TIDE report from saved detections.")
    parser.add_argument("records", type=Path, help="records.json produced by detaudit.run")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tide-cache-dir", type=Path,
                        help="Writable TIDE cache directory; defaults inside --out-dir.")
    parser.add_argument("--plot", action="store_true", help="Also ask TIDE to render its summary PNG.")
    args = parser.parse_args()

    records = json.loads(args.records.read_text(encoding="utf-8"))
    if not records:
        raise RuntimeError("records.json is empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # TIDE otherwise creates a cache in a user-specific home-directory path.
    # Keep it with the run so batch jobs work on restricted compute nodes and
    # so no shared home-directory state changes the analysis.
    tide_cache = args.tide_cache_dir or args.out_dir / ".tide-cache"
    tide_cache.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = args.out_dir / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TIDE_PATH"] = str(tide_cache)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    from tidecv import TIDE, datasets
    ground_truth, detections = coco_payload(records)
    gt_path, pred_path = args.out_dir / "tide_ground_truth.json", args.out_dir / "tide_predictions.json"
    write_json(gt_path, ground_truth)
    write_json(pred_path, detections)

    tide = TIDE()
    # This mirrors COCO's 0.50:0.95 IoU range. TIDE attributes main errors at
    # 0.50 by design, while the threshold sweep supplies its reported mAP.
    tide.evaluate_range(datasets.COCO(str(gt_path)), datasets.COCOResult(str(pred_path)), mode=TIDE.BOX)
    tide.summarize()
    if args.plot:
        tide.plot(str(args.out_dir))
    report = {
        "source_records": str(args.records),
        "records_sha256": sha256(args.records),
        "tide_version": "tidecv",
        "iou_range": list(TIDE.COCO_THRESHOLDS),
        "main_error_dAP": tide.get_main_errors(),
        "special_error_dAP": tide.get_special_errors(),
        "interpretation": {
            "Dupe": "Surviving duplicate detections after the model's configured post-processing; it is not itself proof that NMS caused the error.",
            "NMS_intervention": "Causality is tested separately by the final-stage NMS-IoU sweep with model weights fixed.",
        },
    }
    write_json(args.out_dir / "tide_report.json", report)


if __name__ == "__main__":
    main()
