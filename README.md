# Cross-Domain 2D Object Detection: Accuracy, Robustness, and Adaptation

This repository contains the code, configurations, and exact commands for a
reproducible experimental evaluation of modern 2D object detectors. The
project extends the course `ngdet` tutorial with a controlled study of how
cross-domain adaptation affects both target-domain accuracy and source-domain
retention.

## Research question

For COCO-pretrained detectors deployed in autonomous-driving scenes, which
detector family offers the best accuracy--latency trade-off under domain shift,
and can a small balanced fine-tuning set improve transfer without unacceptable
forgetting of the source domain?

The study will compare three configurations from two families:

| Configuration | Family | Role |
|---|---|---|
| YOLO11s | one-stage, dense prediction | real-time baseline and fine-tuning candidate |
| YOLO11m | one-stage, dense prediction | capacity/speed trade-off |
| RT-DETR-R50 | transformer set prediction | NMS-free architectural comparison |

Evaluation uses COCO val2017 as the in-domain reference and nuImages as the
driving-domain benchmark. Both are projected into a documented common
taxonomy: `vehicle`, `person`, and `cyclist`.

## Experimental design

1. **Zero-shot baseline:** evaluate all three COCO-pretrained checkpoints on
   fixed, deterministic samples from both domains.
2. **Transfer diagnosis:** break AP down by unified class and COCO object scale;
   record end-to-end latency, throughput, and peak GPU memory.
3. **Adaptation ablation:** fine-tune the best latency-constrained model on a
   balanced mixed COCO + nuImages train set, then compare target gain against
   COCO retention. A target-only fine-tune is included as a forgetting control.

The key result is a **Pareto-style transfer profile**, rather than a single
mAP number: target AP, source AP retained, worst-class AP, small-object AP,
latency, and memory are reported together. All choices made after inspecting
the data and results will be recorded in the final report.

## Repository layout

- `configs/` — versioned experiment definitions (no paths or secrets).
- `scripts/` — deterministic setup, evaluation, analysis, and SLURM entry
  points.
- `src/` — project-specific measurement and analysis code.
- `docs/` — taxonomy and reproducibility documentation.
- `commands/` — exact commands used for each completed experiment.

Datasets, upstream repositories, model weights, logs, and generated figures
are deliberately excluded from Git. See `docs/reproducibility.md` before
running an experiment.

## Status

Scaffold and protocol are prepared. HPC module names, storage paths, scheduler
rules, and the available GPU partitions must be supplied before the first run.

## Academic integrity

This repository is a reproducibility record, not a substitute for technical
understanding. Results and explanations will be checked against source code,
papers, and the actual saved experiment metadata.
