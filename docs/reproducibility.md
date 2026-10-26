# Reproducibility protocol

Each experiment is identified by a configuration file and writes a compact
metadata record containing:

- Git commit of this repository and the pinned upstream tutorial commit.
- Checkpoint identifier (and SHA-256 when the checkpoint is a local file).
- Dataset version, split, taxonomy version, and deterministic image manifest.
- Framework, Python, PyTorch, CUDA, and GPU model information.
- Model-native preprocessing, batch size, threshold/NMS settings, and exact
  command. (No random sampling is used by the deterministic evaluation split.)
- Metrics, inference timing methodology, and peak allocated GPU memory.
- Slurm job ID, node/GPU inventory, and the output-file SHA-256 manifest.

Generated data and weights are not committed. The final report will reproduce
the commands and include enough output metadata for an independent rerun.

## Planned execution order

1. Validate the environment and every model adapter on 5--20 images.
2. Generate fixed evaluation manifests for KITTI val and official nuScenes val
   CAM_FRONT images.
3. Run the zero-shot baseline matrix.
4. Run latency/memory profiling after warm-up with batch size 1, on the same
   GPU type for every model.
5. Run final-stage NMS-IoU sweeps for YOLO and Faster R-CNN only.
6. Generate TIDE and scale/density/overlap analyses from saved predictions.
7. Regenerate all tables/figures directly from result JSON files.

No conclusions are pre-registered: negative results will be retained and
discussed if they occur.

## Local smoke testing

The MacBook is appropriate for validating adapters and output artifacts, but
not for the full experimental matrix. Create a local virtual environment,
install the project plus the model runtimes, and use `--device auto`. The
runner selects CUDA when available, then Apple Metal Performance Shaders (MPS),
then CPU. HPC commands remain explicit `--device cuda` commands.

Use a bounded, local dataset copy only for smoke tests. A local smoke run is
not evidence for the final report and must not be mixed with the course-dataset
results.
