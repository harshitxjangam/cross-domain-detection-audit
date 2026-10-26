# Cross-Domain 2D Object Detection: A Post-Processing Failure Audit

This repository contains the code, configurations, and exact commands for a
reproducible experimental evaluation of modern 2D object detectors. The
project extends the course `ngdet` tutorial with a controlled audit of
post-processing and error modes under cross-domain shift.

## Research question

For COCO-pretrained detectors transferred without fine-tuning to two
autonomous-driving domains, how much of the failure on small and nearby objects
is associated with duplicate suppression versus upstream recall, localization,
or classification errors?

The study will compare three configurations from two families:

| Configuration | Family | Role |
|---|---|---|
| YOLO11s | one-stage, dense prediction | NMS-based real-time baseline |
| YOLO11m | one-stage, dense prediction | capacity/speed trade-off |
| RT-DETR-R50 | transformer set prediction | NMS-free architectural comparison |
| Faster R-CNN R50-FPN v2 | two-stage, region-based | anchor/RoI reference |

Evaluation uses COCO val2017 as the source domain and KITTI as the
autonomous-driving target domain. Both use a deliberately strict taxonomy:
`person`, `car`, and `truck`. KITTI `Van`/`Cyclist` and COCO categories outside
the three-class protocol are excluded rather than relabeled.

## Experimental design

1. **Zero-shot baseline:** evaluate all four COCO-pretrained checkpoints on
   the declared COCO val2017 and KITTI validation splits.
2. **Failure strata:** break AP and recall down by class, original-image object
   scale, same-class overlap, and all-class neighborhood density.
3. **NMS intervention:** sweep final-stage NMS IoU only for YOLO and Faster
   R-CNN. For Faster R-CNN, RPN NMS remains fixed; RT-DETR is an NMS-free
   architecture comparison, not an NMS toggle.
4. **Error decomposition:** apply TIDE to saved predictions to measure
   classification, localization, combined, duplicate, background, and missed
   ground-truth contributions.
5. **Speed profile:** record end-to-end latency, throughput, and peak GPU
   memory on the same allocated GPU type.

The key result is a **failure anatomy**, rather than a single mAP number:
target AP, scale/class-conditioned recall, TIDE errors, NMS sensitivity,
latency, and memory are reported together. The study does not claim a new
state-of-the-art detector; it tests a falsifiable architectural hypothesis with
actual saved experiment artifacts.

## Repository layout

- `configs/`: versioned experiment definitions (no paths or secrets).
- `scripts/`: deterministic setup, evaluation, analysis, and SLURM entry
  points.
- `src/`: project-specific measurement and analysis code.
- `docs/`: taxonomy and reproducibility documentation.
- `commands/`: exact commands used for each completed experiment.
- `scripts/`: HPC staging, Slurm submission, and result-generation helpers.

Datasets, upstream repositories, model weights, logs, and generated figures
are deliberately excluded from Git. See `docs/reproducibility.md` before
running an experiment. If the course HPC scheduler association is unavailable,
the bounded single-GPU cloud fallback is documented in
`docs/runpod-operations.md`.

## Local verification

After creating a Python 3.10+ environment and installing the lightweight
project dependencies, run the CPU-only artifact checks before using HPC:

```bash
python -m unittest discover -s tests -v
```

They validate the taxonomy, COCO-format metric schema, final RoI-NMS-only
intervention guard, original-coordinate strata, and saved-records-to-TIDE path.

## Status

The protocol and HPC template are prepared. The exact class storage roots,
available GPU partition, and installed module versions must be verified after
login before the first run.

## Academic integrity

This repository is a reproducibility record, not a substitute for technical
understanding. Results and explanations will be checked against source code,
papers, and the actual saved experiment metadata.
