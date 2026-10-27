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

The execution protocol uses COCO val2017 as the source domain and a
deterministic 1,500-image KITTI holdout as the target-domain pool. The primary
matrix evaluates a fixed 1,000-image subset from each pool. This cap is a
budgeted, reproducible protocol decision, not a claim of full-dataset metrics.
The result metadata records the exact image manifest.

COCO val2017 can be downloaded directly into the Pod:

```bash
cd /workspace/cross-domain-detection-audit
bash scripts/cloud/download_coco_val.sh
```

Download KITTI's labeled object-detection training images and labels only after
accepting its official terms. Extract them under
`/workspace/detaudit-data/kitti`, then create the declared target holdout:

```bash
python scripts/cloud/prepare_kitti_split.py \
  --root /workspace/detaudit-data/kitti --val-count 1500 --seed 2026
```

After the data roots are verified, activate the virtual environment and run the
preflight command with the actual roots:

```bash
source /workspace/venv-detaudit/bin/activate
export PYTHONPATH=/workspace/cross-domain-detection-audit/src
python -m detaudit.preflight \
  --coco-root /workspace/detaudit-data/coco \
  --kitti-root /workspace/detaudit-data/kitti \
  --upstream-root /workspace/DeepDataMiningLearning
```

## Result tables

After the baseline and intervention runs finish, generate the report-ready
summary directly from the saved artifacts:

```bash
python -m detaudit.summarize /workspace/detaudit-outputs \
  --csv /workspace/detaudit-outputs/summary.csv \
  --json /workspace/detaudit-outputs/summary.json
```

The summary retains the model, dataset, image count, AP and AR metrics,
per-class AP, latency, throughput, peak CUDA allocation, NMS threshold, TIDE
errors, and the raw-record SHA-256 identifier.

## Shutdown discipline

Stop the Pod immediately after a completed command or deliberate checkpoint.
Persist only required outputs, environment records, and command logs. Download
the compact result artifacts to the local machine before terminating the Pod.
Never commit dataset files, checkpoints, cloud credentials, SSH keys, or
provider access tokens.
