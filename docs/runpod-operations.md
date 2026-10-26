# RunPod execution notes

Use a single CUDA Pod only for this course project. A 24 GB GPU is sufficient
for batch-size-one inference with the four selected checkpoints. Prefer a
secure, persistent Pod and select a PyTorch CUDA image that includes Jupyter
and SSH. Jupyter is useful for inspecting files and plots; the reproducible
experiments must be launched with the versioned commands in this repository.

## Before deploying

1. Keep the Pod count at one GPU.
2. Select a 24 GB GPU. Do not choose a multi-GPU configuration.
3. Set a spend limit in the provider dashboard if available.
4. Allocate only the storage required for code, model caches, and the selected
   data subset. Do not copy course-shared data to a cloud provider without
   confirmation that the dataset terms and course policy permit it.
5. Record the displayed GPU type, hourly rate, image name, region, disk size,
   and Pod ID in the local execution log.

## Bootstrap

Open a terminal in the Pod and run:

```bash
git clone https://github.com/harshitxjangam/cross-domain-detection-audit.git /workspace/cross-domain-detection-audit
cd /workspace/cross-domain-detection-audit
bash scripts/cloud/bootstrap.sh
```

The bootstrap validates CUDA, creates a workspace virtual environment, clones
the public tutorial repository, and installs the pinned analysis dependencies.
It does not download a dataset or begin an experiment.

## Data and first run

Use only an authorized dataset copy. Start with a small, documented pilot
manifest before downloading or processing a larger validation subset. Do not
mix pilot results with final-report metrics.

After the data root is verified, activate the virtual environment and run the
preflight command with the actual roots:

```bash
source /workspace/venv-detaudit/bin/activate
export PYTHONPATH=/workspace/cross-domain-detection-audit/src
python -m detaudit.preflight \
  --kitti-root /workspace/detaudit-data/kitti \
  --nuscenes-root /workspace/detaudit-data/nuscenes/v1.0-trainval \
  --upstream-root /workspace/DeepDataMiningLearning
```

## Shutdown discipline

Stop the Pod immediately after a completed command or deliberate checkpoint.
Persist only required outputs, environment records, and command logs. Download
the compact result artifacts to the local machine before terminating the Pod.
Never commit dataset files, checkpoints, cloud credentials, SSH keys, or
provider access tokens.
