# Cross-Domain 2D Object Detection Audit

This repository contains a reproducible Track A evaluation of COCO-pretrained
object detectors transferred to the KITTI driving domain without fine-tuning.
It extends the course `ngdet` tutorial with a controlled analysis of domain
shift, post-processing, and detection failures.

## Protocol

- **Models:** YOLO11s, YOLO11m, Faster R-CNN R50-FPN v2, and RT-DETR R50.
- **Detector families:** one-stage dense prediction, two-stage region-based,
  and query-based transformer detection.
- **Domains:** COCO val2017 and KITTI.
- **Shared taxonomy:** `person`, `car`, and `truck`.
- **Evaluation:** deterministic 1,000-image subsets, COCO-style metrics,
  TIDE error decomposition, latency, FPS, and peak CUDA memory.
- **NMS intervention:** final-stage IoU sweep from 0.30 to 0.90 for YOLO11s
  and Faster R-CNN, with Faster R-CNN RPN NMS held fixed.

## Main finding

Relaxing NMS selectively improves recall for same-class overlapping objects,
but an overly permissive threshold increases duplicate errors and reduces
aggregate mAP. The observed NMS gain is small relative to the COCO-to-KITTI
domain shift, indicating that localization and background-related failures
remain important.

## Repository contents

- `report/`: final technical report and figures.
- `results/`: compact summaries of all completed runs and the NMS mechanism
  analysis.
- `src/detaudit/`: evaluation, taxonomy, TIDE, density-strata, and NMS
  mechanism analysis code.
- `configs/`: versioned experiment definitions.
- `scripts/`: cloud setup, evaluation, and report-generation scripts.
- `commands/`: exact completed experiment command patterns.
- `docs/`: taxonomy, reproducibility, upstream, and runtime documentation.
- `tests/`: CPU-only checks for evaluation and analysis behavior.

## Reproduce the report

The report is generated from saved local experiment artifacts, which are not
committed because they include large raw predictions and logs. After restoring
`artifacts/final_evidence/`, run:

```bash
python3 -m venv .report-venv
.report-venv/bin/python -m pip install matplotlib reportlab pypdf pillow
.report-venv/bin/python scripts/analysis/build_submission.py
```

Run project checks with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

See [the final report](report/cross_domain_detection_audit.pdf) for the full
experimental protocol, quantitative results, limitations, and references.
