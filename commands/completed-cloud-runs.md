# Completed cloud runs

This log records the command pattern used for the submitted evidence. It has no
credentials, private paths, or provider identifiers. Each run wrote a
`metrics.json`, `records.json`, `strata.json`, `manifest.json`, console log,
and TIDE report; the SHA-256 of each raw record file is in `results/summary.csv`.

## Runtime

- Python 3.12.3, PyTorch 2.8.0+cu128, CUDA runtime 12.8.
- `pycocotools` 2.0.11, `tidecv` 1.0.1, `ultralytics` 8.3.0, and
  `transformers` 4.45.2.
- Baseline matrix: one NVIDIA RTX A6000 48 GB GPU.
- Recovered NMS points: one NVIDIA GeForce RTX 4090 24 GB GPU.
- Batch size 1; deterministic 1,000-image manifests; score floor 0.001;
  saved-candidate cap 300; COCO metric cap 100.

## Baseline matrix

The following eight exact top-level invocations were run once each. The
environment variables use the cloud workspace defaults encoded by the project
scripts.

```bash
MAX_IMAGES=1000 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh yolo11s-coco-1000 yolo:yolo11s.pt coco
MAX_IMAGES=1000 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh yolo11s-kitti-1000 yolo:yolo11s.pt kitti
MAX_IMAGES=1000 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh yolo11m-coco-1000 yolo:yolo11m.pt coco
MAX_IMAGES=1000 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh yolo11m-kitti-1000 yolo:yolo11m.pt kitti
MAX_IMAGES=1000 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh fasterrcnn-r50-fpn-v2-coco-1000 torchvision:fasterrcnn_resnet50_fpn_v2 coco
MAX_IMAGES=1000 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh fasterrcnn-r50-fpn-v2-kitti-1000 torchvision:fasterrcnn_resnet50_fpn_v2 kitti
MAX_IMAGES=1000 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh rtdetr-r50-coco-1000 hf_detr:PekingU/rtdetr_r50vd coco
MAX_IMAGES=1000 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh rtdetr-r50-kitti-1000 hf_detr:PekingU/rtdetr_r50vd kitti
```

Each wrapper expands to `python -m detaudit.run` with `--split val`,
`--score-floor 0.001`, `--max-detections 300`, `--max-images 1000`,
`--yolo-imgsz 640`, and `--device cuda`. The fixed `Driving3Taxonomy`
(`person`, `car`, `truck`) is constructed in the runner rather than supplied
through a CLI flag. Metrics use a fixed COCO `maxDets=100` cap.

## Final-stage NMS intervention

The following exact top-level invocations were run for NMS IoU values `0.30`,
`0.45`, `0.60`, `0.75`, and `0.90` on KITTI. Model weights, image manifest,
score floor, and candidate cap were fixed. For Faster R-CNN, only final RoI
detection NMS was changed; RPN NMS was held fixed.

```bash
MAX_IMAGES=1000 NMS_IOU=0.3 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh yolo11s-kitti-nms-03 yolo:yolo11s.pt kitti
MAX_IMAGES=1000 NMS_IOU=0.45 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh yolo11s-kitti-nms-045 yolo:yolo11s.pt kitti
MAX_IMAGES=1000 NMS_IOU=0.6 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh yolo11s-kitti-nms-06 yolo:yolo11s.pt kitti
MAX_IMAGES=1000 NMS_IOU=0.75 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh yolo11s-kitti-nms-075 yolo:yolo11s.pt kitti
MAX_IMAGES=1000 NMS_IOU=0.9 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh yolo11s-kitti-nms-09 yolo:yolo11s.pt kitti

MAX_IMAGES=1000 NMS_IOU=0.3 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh fasterrcnn-kitti-nms-03 torchvision:fasterrcnn_resnet50_fpn_v2 kitti
MAX_IMAGES=1000 NMS_IOU=0.45 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh fasterrcnn-kitti-nms-045 torchvision:fasterrcnn_resnet50_fpn_v2 kitti
MAX_IMAGES=1000 NMS_IOU=0.6 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh fasterrcnn-kitti-nms-06 torchvision:fasterrcnn_resnet50_fpn_v2 kitti
MAX_IMAGES=1000 NMS_IOU=0.75 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh fasterrcnn-kitti-nms-075 torchvision:fasterrcnn_resnet50_fpn_v2 kitti
MAX_IMAGES=1000 NMS_IOU=0.9 OUTPUT_ROOT=/workspace/detaudit-outputs bash scripts/cloud/run_evaluation.sh fasterrcnn-kitti-nms-09 torchvision:fasterrcnn_resnet50_fpn_v2 kitti
```

The wrapper appends `--nms-iou` with the supplied `NMS_IOU` value. The runner
sets `detector.model.roi_heads.nms_thresh` for Faster R-CNN and records
`rpn_nms_changed: false` in the run metadata.

`RT-DETR` was not included in this intervention because it uses query-based,
set-prediction inference rather than a final NMS stage.
