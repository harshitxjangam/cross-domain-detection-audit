#!/usr/bin/env bash
# Submit with: sbatch scripts/hpc/run_audit.sbatch <run-name> <model> <dataset> [nms-iou]
# This is a single-GPU job by course policy. Set paths only in local hpc.env.
#SBATCH --job-name=detaudit
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:p100:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

RUN_NAME=${1:?run name required}
MODEL=${2:?model spec required}
DATASET=${3:?dataset required}
NMS_IOU=${4:-}
MAX_IMAGES=${MAX_IMAGES:-0}
PROJECT_ROOT=${PROJECT_ROOT:?export PROJECT_ROOT before sbatch}
UPSTREAM_ROOT=${UPSTREAM_ROOT:?export UPSTREAM_ROOT before sbatch}
PROJECT_DATA_ROOT=${PROJECT_DATA_ROOT:?export PROJECT_DATA_ROOT before sbatch}
PROJECT_SCRATCH_ROOT=${PROJECT_SCRATCH_ROOT:?export PROJECT_SCRATCH_ROOT before sbatch}

if [[ -n "${CUDA_MODULE:-}" && "${CUDA_MODULE}" != "<leave-unset-unless-a-package-needs-nvcc>" ]]; then
  module load "${CUDA_MODULE}"
fi
source "${VENV_PATH:?export VENV_PATH before sbatch}/bin/activate"
export HF_HOME="${PROJECT_CACHE_ROOT:?export PROJECT_CACHE_ROOT before sbatch}/huggingface"
export TORCH_HOME="${PROJECT_CACHE_ROOT}/torch"
export PYTHONPATH="${PROJECT_ROOT}/src:${UPSTREAM_ROOT}:${PYTHONPATH:-}"

KITTI_ROOT="${PROJECT_DATA_ROOT}/datasets/kitti"
COCO_ROOT="${PROJECT_DATA_ROOT}/datasets/coco"
NUSCENES_ROOT="${PROJECT_DATA_ROOT}/datasets/nuscenes/v1.0-trainval"
OUT_DIR="${PROJECT_SCRATCH_ROOT}/runs/${RUN_NAME}"

mkdir -p "${OUT_DIR}"
CMD=(python -m detaudit.run --upstream-root "${UPSTREAM_ROOT}" --model "${MODEL}" --dataset "${DATASET}" --out-dir "${OUT_DIR}" --score-floor 0.001 --max-detections 300 --max-images "${MAX_IMAGES}" --yolo-imgsz 640 --device cuda)
if [[ "${DATASET}" == "coco" ]]; then
  CMD+=(--root "${COCO_ROOT}" --split val)
elif [[ "${DATASET}" == "kitti" ]]; then
  CMD+=(--root "${KITTI_ROOT}" --split val)
elif [[ "${DATASET}" == "nuscenes" ]]; then
  CMD+=(--root "${NUSCENES_ROOT}" --split val --camera CAM_FRONT)
else
  echo "Unsupported dataset: ${DATASET}. Choose coco, kitti, or nuscenes." >&2
  exit 2
fi
if [[ -n "${NMS_IOU}" ]]; then CMD+=(--nms-iou "${NMS_IOU}"); fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-interactive}"
nvidia-smi
printf 'COMMAND: '; printf '%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}" | tee "${OUT_DIR}/console.log"
python -m detaudit.analyze "${OUT_DIR}/records.json" --out "${OUT_DIR}/strata.json"
python -m detaudit.tide_report "${OUT_DIR}/records.json" --out-dir "${OUT_DIR}/tide"
