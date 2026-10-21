# Execution plan

Every GPU request is a separate, single-GPU Slurm job. Do not submit the full
matrix until the previous stage has produced valid artifacts.

## 1. Stage and validate

On the HPC head node, clone the upstream repository and this project into the
student directory, create a local environment file from
`scripts/hpc/hpc.env.example`, and cache checkpoints before GPU jobs if the
network policy permits. Record the upstream commit SHA.

Submit `scripts/hpc/preflight.sbatch` first. It is a CPU-only job which confirms
the shared environment plus one sample from KITTI val and official nuScenes val
CAM_FRONT without writing to shared data or reserving a GPU.

## 2. Smoke test

Run `configs/experiments/pilot_yolo11s_kitti.json` on 20 images. It must create:

- `records.json` — all GT and post-NMS detections;
- `manifest.json` — deterministic evaluated image IDs;
- `metrics.json` — COCO metrics, timing, GPU memory, versions, model settings;
- `strata.json` — original-coordinate scale and density counts;
- `tide/tide_report.json` — error decomposition linked by the `records.json`
  SHA-256, plus the exact COCO-format inputs TIDE consumed;
- Slurm stdout/stderr and `console.log`.

Inspect one record manually before continuing.

## 3. Baseline matrix

For each model-dataset pair from `baseline_matrix.json`, submit one bounded job.
Record the job ID in `commands/`. Use native checkpoint preprocessing. All runs
use the same strict three-class taxonomy, `score_floor=0.001`, retain 300 detections in
artifacts, and report standard COCO `maxDets=100` metrics.

## 4. Intervention

After baseline results are verified, submit only the YOLO11s and Faster R-CNN
nuScenes NMS jobs listed in `nms_sweep.json`. For Faster R-CNN, the script
changes only the final RoI-head NMS threshold; RPN NMS is unchanged. RT-DETR is
not part of this sweep because it is NMS-free.

## 5. Error analysis and report

Run TIDE from saved artifacts and verify the required `Cls`, `Loc`, `Both`,
`Dupe`, `Bkg`, and `Miss` interpretation. Use the project strata to distinguish
same-class overlap (directly NMS-relevant) from all-class neighborhood density.
Generate tables/figures only from recorded JSON artifacts.
