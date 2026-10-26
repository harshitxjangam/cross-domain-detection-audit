#!/usr/bin/env bash
# Prepare a RunPod workspace from the public project repository.
# Run inside a CUDA-enabled PyTorch RunPod Pod, not on the local Mac.

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
PROJECT_ROOT="${PROJECT_ROOT:-${WORKSPACE_ROOT}/cross-domain-detection-audit}"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-${WORKSPACE_ROOT}/DeepDataMiningLearning}"
VENV_PATH="${VENV_PATH:-${WORKSPACE_ROOT}/venv-detaudit}"
PROJECT_REPOSITORY="${PROJECT_REPOSITORY:-https://github.com/harshitxjangam/cross-domain-detection-audit.git}"
UPSTREAM_REPOSITORY="${UPSTREAM_REPOSITORY:-https://github.com/lkk688/DeepDataMiningLearning.git}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "A CUDA RunPod image is required: nvidia-smi was not found." >&2
  exit 1
fi

nvidia-smi

if [[ ! -d "${PROJECT_ROOT}/.git" ]]; then
  git clone "${PROJECT_REPOSITORY}" "${PROJECT_ROOT}"
else
  git -C "${PROJECT_ROOT}" pull --ff-only
fi

if [[ ! -d "${UPSTREAM_ROOT}/.git" ]]; then
  git clone "${UPSTREAM_REPOSITORY}" "${UPSTREAM_ROOT}"
fi

python -m venv --system-site-packages "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_ROOT}/requirements-hpc.txt"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see a CUDA GPU in this Pod.")

print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
})
PY

mkdir -p "${WORKSPACE_ROOT}/detaudit-data" "${WORKSPACE_ROOT}/detaudit-outputs"
printf 'PROJECT_ROOT=%s\nUPSTREAM_ROOT=%s\nVENV_PATH=%s\n' \
  "${PROJECT_ROOT}" "${UPSTREAM_ROOT}" "${VENV_PATH}"
