#!/usr/bin/env bash
# CPU-only dataset/environment check. Submit from HPC after sourcing local hpc.env:
# sbatch scripts/hpc/preflight.sbatch
# It never requests a GPU and never writes to shared datasets.
#SBATCH --job-name=detaudit-preflight
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=preflight-%j.out
#SBATCH --error=preflight-%j.err

set -euo pipefail

: "${PROJECT_ROOT:?source local hpc.env before sbatch}"
: "${PROJECT_DATA_ROOT:?source local hpc.env before sbatch}"
: "${VENV_PATH:?source local hpc.env before sbatch}"

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
python -m detaudit.preflight \
  --kitti-root "${PROJECT_DATA_ROOT}/datasets/kitti" \
  --nuscenes-root "${PROJECT_DATA_ROOT}/datasets/nuscenes/v1.0-trainval" \
  --upstream-root "${UPSTREAM_ROOT:?source local hpc.env before sbatch}"
