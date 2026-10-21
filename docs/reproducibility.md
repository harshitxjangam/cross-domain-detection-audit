# Reproducibility protocol

Each experiment is identified by a configuration file and writes a compact
metadata record containing:

- Git commit of this repository and the pinned upstream tutorial commit.
- Checkpoint identifier and SHA-256 when available.
- Dataset version, split, taxonomy version, and deterministic image manifest.
- Framework, Python, PyTorch, CUDA, and GPU model information.
- Resolution, batch size, threshold/NMS settings, random seed, and command.
- Metrics, inference timing methodology, and peak allocated GPU memory.

Generated data and weights are not committed. The final report will reproduce
the commands and include enough output metadata for an independent rerun.

## Planned execution order

1. Validate the environment and every model adapter on 5--20 images.
2. Generate fixed evaluation manifests for COCO val2017 and nuImages.
3. Run the zero-shot matrix.
4. Run latency/memory profiling after warm-up, using fixed batch size and
   resolution.
5. Create leakage-free target train/validation/test splits and run the two
   adaptation conditions.
6. Regenerate all tables/figures directly from result JSON files.

No conclusions are pre-registered: negative results will be retained and
discussed if they occur.
