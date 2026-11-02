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

The baseline matrix was run once for each model and dataset pair:

```bash
MAX_IMAGES=1000 bash scripts/cloud/run_evaluation.sh \
  <run-name> <model-id> <coco-or-kitti>
```

| Run name | Model ID | Dataset |
|---|---|---|
| `yolo11s-coco-1000` | `yolo:yolo11s.pt` | COCO val2017 |
| `yolo11s-kitti-1000` | `yolo:yolo11s.pt` | KITTI |
| `yolo11m-coco-1000` | `yolo:yolo11m.pt` | COCO val2017 |
| `yolo11m-kitti-1000` | `yolo:yolo11m.pt` | KITTI |
| `fasterrcnn-r50-fpn-v2-coco-1000` | `torchvision:fasterrcnn_resnet50_fpn_v2` | COCO val2017 |
| `fasterrcnn-r50-fpn-v2-kitti-1000` | `torchvision:fasterrcnn_resnet50_fpn_v2` | KITTI |
| `rtdetr-r50-coco-1000` | `hf_detr:PekingU/rtdetr_r50vd` | COCO val2017 |
| `rtdetr-r50-kitti-1000` | `hf_detr:PekingU/rtdetr_r50vd` | KITTI |

## Final-stage NMS intervention

The following pattern was run for NMS IoU values `0.30`, `0.45`, `0.60`,
`0.75`, and `0.90` on KITTI. Model weights, image manifest, score floor, and
candidate cap were fixed. For Faster R-CNN, only final RoI detection NMS was
changed; RPN NMS was held fixed.

```bash
MAX_IMAGES=1000 NMS_IOU=<threshold> \
  bash scripts/cloud/run_evaluation.sh <run-name> <model-id> kitti
```

| Run prefix | Model ID |
|---|---|
| `yolo11s-kitti-nms-{03,045,06,075,09}` | `yolo:yolo11s.pt` |
| `fasterrcnn-kitti-nms-{03,045,06,075,09}` | `torchvision:fasterrcnn_resnet50_fpn_v2` |

`RT-DETR` was not included in this intervention because it uses query-based,
set-prediction inference rather than a final NMS stage.
