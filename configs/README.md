# Experiment configurations

This directory will contain one human-readable configuration per executed run.
Configurations must use relative paths or environment-variable placeholders;
do not commit personal paths, credentials, or dataset contents.

Planned names:

- `experiments/pilot_yolo11s_kitti.json`
- `experiments/baseline_matrix.json`
- `experiments/nms_sweep.json`

The declared protocol is COCO-pretrained zero-shot evaluation on the shared
KITTI and nuScenes datasets, not nuImages (which is unavailable on the course
storage).
