#!/usr/bin/env bash
# Submit with: sbatch scripts/hpc/run_audit.sbatch <run-name> <model> <dataset> [nms-iou]
# This is a single-GPU job by course policy. Set paths only in local hpc.env.
#SBATCH --job-name=detaudit
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

RUN_NAME=${1:?run name required}
MODEL=${2:?model spec required}
DATASET=${3:?dataset required}
NMS_IOU=${4:-}
PROJECT_ROOT=${PROJECT_ROOT:?export PROJECT_ROOT before sbatch}
UPSTREAM_ROOT=${UPSTREAM_ROOT:?export UPSTREAM_ROOT before sbatch}
PROJECT_DATA_ROOT=${PROJECT_DATA_ROOT:?export PROJECT_DATA_ROOT before sbatch}
PROJECT_SCRATCH_ROOT=${PROJECT_SCRATCH_ROOT:?export PROJECT_SCRATCH_ROOT before sbatch}

module load "${CUDA_MODULE:?export CUDA_MODULE before sbatch}"
source "${CONDA_BASE:?export CONDA_BASE before sbatch}/bin/activate" "${CONDA_ENV:?export CONDA_ENV before sbatch}"
export HF_HOME="${PROJECT_CACHE_ROOT:?export PROJECT_CACHE_ROOT before sbatch}/huggingface"
export TORCH_HOME="${PROJECT_CACHE_ROOT}/torch"
export PYTHONPATH="${PROJECT_ROOT}/src:${UPSTREAM_ROOT}:${PYTHONPATH:-}"

COCO_ROOT="${PROJECT_DATA_ROOT}/datasets/coco2017/val2017"
COCO_ANN="${PROJECT_DATA_ROOT}/datasets/coco2017/annotations/instances_val2017.json"
NUIMAGES_ROOT="${PROJECT_DATA_ROOT}/datasets/nuimages"
OUT_DIR="${PROJECT_SCRATCH_ROOT}/runs/${RUN_NAME}"

mkdir -p "${OUT_DIR}"
CMD=(python -m detaudit.run --upstream-root "${UPSTREAM_ROOT}" --model "${MODEL}" --dataset "${DATASET}" --out-dir "${OUT_DIR}" --score-floor 0.001 --max-detections 300 --yolo-imgsz 640 --device cuda)
if [[ "${DATASET}" == "coco" ]]; then
  CMD+=(--root "${COCO_ROOT}" --coco-ann "${COCO_ANN}")
else
  CMD+=(--root "${NUIMAGES_ROOT}" --nuimages-version v1.0-val)
fi
if [[ -n "${NMS_IOU}" ]]; then CMD+=(--nms-iou "${NMS_IOU}"); fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-interactive}"
nvidia-smi
printf 'COMMAND: '; printf '%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}" | tee "${OUT_DIR}/console.log"
python -m detaudit.analyze "${OUT_DIR}/records.json" --out "${OUT_DIR}/strata.json"
python -m detaudit.tide_report "${OUT_DIR}/records.json" --out-dir "${OUT_DIR}/tide"
