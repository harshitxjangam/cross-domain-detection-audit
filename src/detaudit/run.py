"""Run one reproducible model × dataset evaluation using the course harness.

The module deliberately records post-NMS model output. It is not a replacement
for the upstream harness; it adds the experiment controls and artifacts needed
for this project's NMS audit.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .datasets import build_dataset
from .records import sample_record, sha256, write_json, write_manifest
from .taxonomy import Driving3Taxonomy


COCO_METRIC_KEYS = ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large")


def upstream_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


def configure_detector(detector, nms_iou: float | None, max_detections: int, yolo_imgsz: int):
    """Apply only documented final-stage controls for the audit.

    Faster R-CNN's RPN NMS is untouched. RT-DETR has no NMS control and is used
    only as an architectural comparator.
    """
    postprocessing = {"final_nms_iou": None, "nms_applied": False}
    if detector.family == "torchvision":
        if hasattr(detector.model, "roi_heads"):
            detector.model.roi_heads.score_thresh = detector.score_thr
            detector.model.roi_heads.detections_per_img = max_detections
            if nms_iou is not None:
                detector.model.roi_heads.nms_thresh = nms_iou
            postprocessing = {"final_nms_iou": float(detector.model.roi_heads.nms_thresh),
                              "nms_applied": True, "rpn_nms_changed": False}
    if detector.family == "yolo":
        resolved_nms_iou = nms_iou if nms_iou is not None else 0.7

        def yolo_predict(image, prompt=None):
            results = detector.model.predict(
                image, device=detector.device, conf=0.0, iou=resolved_nms_iou, imgsz=yolo_imgsz,
                max_det=max_detections, verbose=False, half=detector._half,
            )[0]
            boxes = results.boxes.xyxy.detach().cpu().numpy().reshape(-1, 4)
            scores = results.boxes.conf.detach().cpu().numpy().reshape(-1)
            labels = results.boxes.cls.detach().cpu().numpy().astype(int).reshape(-1)
            return detector._fold_to_unified(boxes, scores, labels)

        detector.predict = yolo_predict
        postprocessing = {"final_nms_iou": resolved_nms_iou, "nms_applied": True}
    return detector, postprocessing


def preprocessing_metadata(detector, yolo_imgsz: int) -> dict:
    """Expose each checkpoint's intended resize policy in the result artifact."""
    if detector.family == "yolo":
        return {"type": "letterbox", "input_size": [yolo_imgsz, yolo_imgsz]}
    if detector.family == "torchvision" and hasattr(detector.model, "transform"):
        transform = detector.model.transform
        return {"type": "GeneralizedRCNNTransform", "min_size": list(transform.min_size),
                "max_size": transform.max_size}
    if detector.family == "hf_detr":
        return {"type": "checkpoint image processor", "size": getattr(detector.processor, "size", None)}
    return {"type": "model-native upstream adapter"}


def coco_metrics(records: list[dict], taxonomy: Driving3Taxonomy) -> dict:
    """Compute canonical COCO metrics from saved JSON, independently of a model."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, annotations, detections = [], [], []
    ann_id = 1
    for record in records:
        image_id = int(record["image_id"])
        images.append({"id": image_id, "width": record["width"], "height": record["height"]})
        for box, label in zip(record["gt_boxes_xyxy"], record["gt_labels"]):
            x1, y1, x2, y2 = map(float, box); width, height = x2 - x1, y2 - y1
            annotations.append({"id": ann_id, "image_id": image_id, "category_id": int(label) + 1,
                                "bbox": [x1, y1, width, height], "area": width * height, "iscrowd": 0})
            ann_id += 1
        for box, score, label in zip(record["pred_boxes_xyxy"], record["pred_scores"], record["pred_labels"]):
            x1, y1, x2, y2 = map(float, box)
            detections.append({"image_id": image_id, "category_id": int(label) + 1,
                               "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    cats = [{"id": idx + 1, "name": name} for idx, name in enumerate(taxonomy.classes)]
    gt = COCO(); gt.dataset = {"images": images, "annotations": annotations, "categories": cats}; gt.createIndex()
    if not detections:
        return {**{key: 0.0 for key in COCO_METRIC_KEYS},
                "per_class": {c: 0.0 for c in taxonomy.classes}, "coco_stats": [0.0] * 12}
    evaluator = COCOeval(gt, gt.loadRes(detections), "bbox")
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    precision = evaluator.eval["precision"]
    per_class = {}
    for idx, class_name in enumerate(taxonomy.classes):
        values = precision[:, :, idx, 0, -1]; values = values[values > -1]
        per_class[class_name] = float(values.mean()) if values.size else 0.0
    stats = [float(x) for x in evaluator.stats]
    return {"mAP": stats[0], "AP50": stats[1], "AP75": stats[2], "AP_small": stats[3],
            "AP_medium": stats[4], "AP_large": stats[5], "AR_1": stats[6], "AR_10": stats[7],
            "AR_100": stats[8], "AR_small": stats[9], "AR_medium": stats[10], "AR_large": stats[11],
            "per_class": per_class, "coco_stats": stats}


def environment_metadata(upstream_root: Path) -> dict:
    import torch
    project_root = Path(__file__).resolve().parents[2]
    metadata = {"python": sys.version, "platform": platform.platform(), "hostname": platform.node(), "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                "upstream_commit": upstream_commit(upstream_root), "project_commit": upstream_commit(project_root),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_node": os.environ.get("SLURMD_NODENAME"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if torch.cuda.is_available():
        metadata["gpu"] = torch.cuda.get_device_name(0)
        metadata["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Track-A audit evaluation.")
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model", required=True, help="e.g. yolo:yolo11s.pt or hf_detr:PekingU/rtdetr_r50vd")
    parser.add_argument("--dataset", choices=("kitti", "nuscenes"), required=True)
    parser.add_argument("--root", type=Path, required=True, help="KITTI root or nuScenes v1.0-trainval root")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="val", help="KITTI split or nuScenes official split (val only)")
    parser.add_argument("--camera", default="CAM_FRONT", help="nuScenes camera; fixed to CAM_FRONT by protocol")
    parser.add_argument("--max-images", type=int, default=0, help="0 evaluates the full split")
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--nms-iou", type=float, default=None)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-images", type=int, default=3)
    args = parser.parse_args()
    if args.nms_iou is not None and not 0.0 < args.nms_iou < 1.0:
        parser.error("--nms-iou must be strictly between 0 and 1")
    sys.path.insert(0, str(args.upstream_root))
    from DeepDataMiningLearning.ngdet.detectors.base import build_detector

    taxonomy = Driving3Taxonomy()
    dataset = build_dataset(args.dataset, args.root, taxonomy, split=args.split,
                            camera=args.camera, max_images=args.max_images or None)
    if not len(dataset):
        raise RuntimeError("Selected dataset yielded no images.")
    detector, postprocessing = configure_detector(
        build_detector(args.model, taxonomy, device=args.device, score_thr=args.score_floor),
        args.nms_iou, args.max_detections, args.yolo_imgsz)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Warm-up is intentionally outside timing and never enters metric artifacts.
    for _ in range(args.warmup_images):
        detector.predict(dataset[0].image)
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    records, elapsed = [], []
    for sample in dataset:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.perf_counter(); detection = detector.predict(sample.image)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
        records.append(sample_record(sample, detection, args.dataset))
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    record_path = args.out_dir / "records.json"
    write_json(record_path, records); write_manifest(args.out_dir / "manifest.json", records)
    metrics = coco_metrics(records, taxonomy)
    metadata = environment_metadata(args.upstream_root)
    metadata.update({"model": args.model, "dataset": args.dataset, "root": str(args.root),
                     "num_images": len(records), "score_floor": args.score_floor,
                     "taxonomy": {"name": taxonomy.name, "classes": list(taxonomy.classes)},
                     "dataset_metadata": dataset.metadata,
                     "max_detections_saved": args.max_detections, "coco_max_detections": 100,
                     "postprocessing": postprocessing,
                     "preprocessing": preprocessing_metadata(detector, args.yolo_imgsz), "batch_size": 1,
                     "warmup_images": args.warmup_images, "command": [sys.executable, "-m", "detaudit.run", *sys.argv[1:]],
                     "mean_latency_ms": 1000 * float(np.mean(elapsed)),
                     "median_latency_ms": 1000 * float(np.median(elapsed)), "fps": float(1 / np.mean(elapsed)),
                     "peak_cuda_memory_bytes": peak, "records_sha256": sha256(record_path)})
    write_json(args.out_dir / "metrics.json", {"metrics": metrics, "metadata": metadata})
    print(json.dumps({"metrics": metrics, "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
