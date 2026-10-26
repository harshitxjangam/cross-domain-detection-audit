#!/usr/bin/env bash
# Run exactly one model-dataset pair and generate all reportable artifacts.

set -euo pipefail

RUN_NAME=${1:?run name required}
MODEL=${2:?model spec required}
DATASET=${3:?coco or kitti required}
MAX_IMAGES=${MAX_IMAGES:-1000}
NMS_IOU=${NMS_IOU:-}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace}
PROJECT_ROOT=${PROJECT_ROOT:-${WORKSPACE_ROOT}/cross-domain-detection-audit}
UPSTREAM_ROOT=${UPSTREAM_ROOT:-${WORKSPACE_ROOT}/DeepDataMiningLearning}
VENV_PATH=${VENV_PATH:-${WORKSPACE_ROOT}/venv-detaudit}
DATA_ROOT=${DATA_ROOT:-${WORKSPACE_ROOT}/detaudit-data}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORKSPACE_ROOT}/detaudit-outputs}

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src:${UPSTREAM_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${WORKSPACE_ROOT}/hf-cache"
export TORCH_HOME="${WORKSPACE_ROOT}/torch-cache"

case "${DATASET}" in
  coco) ROOT="${DATA_ROOT}/coco" ;;
  kitti) ROOT="${DATA_ROOT}/kitti" ;;
  *) echo "DATASET must be coco or kitti" >&2; exit 2 ;;
esac

OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUT_DIR}"
COMMAND=(python -m detaudit.run --upstream-root "${UPSTREAM_ROOT}" --model "${MODEL}"
  --dataset "${DATASET}" --root "${ROOT}" --out-dir "${OUT_DIR}" --split val
  --score-floor 0.001 --max-detections 300 --max-images "${MAX_IMAGES}"
  --yolo-imgsz 640 --device cuda)
if [[ -n "${NMS_IOU}" ]]; then COMMAND+=(--nms-iou "${NMS_IOU}"); fi

printf 'COMMAND: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
"${COMMAND[@]}" | tee "${OUT_DIR}/console.log"
python -m detaudit.analyze "${OUT_DIR}/records.json" --out "${OUT_DIR}/strata.json"
python -m detaudit.tide_report "${OUT_DIR}/records.json" --out-dir "${OUT_DIR}/tide"
