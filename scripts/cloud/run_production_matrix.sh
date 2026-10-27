#!/usr/bin/env bash
# Run the complete Track A matrix and package its evidence in one GPU session.

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
PROJECT_ROOT="${PROJECT_ROOT:-${WORKSPACE_ROOT}/cross-domain-detection-audit}"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-${WORKSPACE_ROOT}/DeepDataMiningLearning}"
VENV_PATH="${VENV_PATH:-${WORKSPACE_ROOT}/venv-detaudit}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/detaudit-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/detaudit-outputs}"
PILOT_ROOT="${PILOT_ROOT:-${WORKSPACE_ROOT}/detaudit-pilots}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-${WORKSPACE_ROOT}/detaudit-evidence}"
LOG_PATH="${WORKSPACE_ROOT}/detaudit-production.log"

mkdir -p "${OUTPUT_ROOT}" "${PILOT_ROOT}" "${EVIDENCE_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

on_error() {
  local exit_code=$?
  trap - ERR
  printf 'failed\n' > "${EVIDENCE_ROOT}/status.txt"
  echo "Production matrix failed with exit code ${exit_code} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Keep the Pod running only while diagnosing or resuming the failed command."
  exit "${exit_code}"
}
trap on_error ERR

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src:${UPSTREAM_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${WORKSPACE_ROOT}/hf-cache"
export TORCH_HOME="${WORKSPACE_ROOT}/torch-cache"

echo "Production matrix started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi
df -h "${WORKSPACE_ROOT}"

python -m detaudit.preflight \
  --coco-root "${DATA_ROOT}/coco" \
  --kitti-root "${DATA_ROOT}/kitti" \
  --upstream-root "${UPSTREAM_ROOT}"

run_one() {
  local run_name=$1
  local model=$2
  local dataset=$3
  local nms_iou=${4:-}
  local out_dir="${OUTPUT_ROOT}/${run_name}"

  if [[ -s "${out_dir}/metrics.json" && -s "${out_dir}/records.json" \
      && -s "${out_dir}/strata.json" && -s "${out_dir}/tide/tide_report.json" ]]; then
    echo "SKIP complete run: ${run_name}"
    return
  fi

  echo "RUN ${run_name}"
  if [[ -n "${nms_iou}" ]]; then
    MAX_IMAGES=1000 NMS_IOU="${nms_iou}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
      bash "${PROJECT_ROOT}/scripts/cloud/run_evaluation.sh" \
      "${run_name}" "${model}" "${dataset}"
  else
    MAX_IMAGES=1000 OUTPUT_ROOT="${OUTPUT_ROOT}" \
      bash "${PROJECT_ROOT}/scripts/cloud/run_evaluation.sh" \
      "${run_name}" "${model}" "${dataset}"
  fi
}

# A separate 20-image target-domain pilot catches data or adapter problems
# before the full matrix. It is not included in the primary result summary.
if [[ ! -s "${PILOT_ROOT}/yolo11s-kitti-20/metrics.json" ]]; then
  MAX_IMAGES=20 OUTPUT_ROOT="${PILOT_ROOT}" \
    bash "${PROJECT_ROOT}/scripts/cloud/run_evaluation.sh" \
    yolo11s-kitti-20 yolo:yolo11s.pt kitti
fi

for dataset in coco kitti; do
  run_one "yolo11s-${dataset}-1000" yolo:yolo11s.pt "${dataset}"
  run_one "yolo11m-${dataset}-1000" yolo:yolo11m.pt "${dataset}"
  run_one "fasterrcnn-r50-fpn-v2-${dataset}-1000" \
    torchvision:fasterrcnn_resnet50_fpn_v2 "${dataset}"
  run_one "rtdetr-r50-${dataset}-1000" hf_detr:PekingU/rtdetr_r50vd "${dataset}"
done

for threshold in 0.3 0.45 0.6 0.75 0.9; do
  suffix=${threshold/./}
  run_one "yolo11s-kitti-nms-${suffix}" yolo:yolo11s.pt kitti "${threshold}"
  run_one "fasterrcnn-kitti-nms-${suffix}" \
    torchvision:fasterrcnn_resnet50_fpn_v2 kitti "${threshold}"
done

python -m detaudit.summarize "${OUTPUT_ROOT}" \
  --csv "${OUTPUT_ROOT}/summary.csv" \
  --json "${OUTPUT_ROOT}/summary.json"

python - "${OUTPUT_ROOT}/summary.json" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1], encoding="utf-8"))
if len(rows) != 18:
    raise SystemExit(f"Expected 18 production runs, found {len(rows)}")
print({"validated_production_runs": len(rows)})
PY

python -m pip freeze > "${EVIDENCE_ROOT}/pip-freeze.txt"
nvidia-smi -q > "${EVIDENCE_ROOT}/nvidia-smi.txt"
git -C "${PROJECT_ROOT}" rev-parse HEAD > "${EVIDENCE_ROOT}/project-commit.txt"
git -C "${UPSTREAM_ROOT}" rev-parse HEAD > "${EVIDENCE_ROOT}/upstream-commit.txt"
cp "${LOG_PATH}" "${EVIDENCE_ROOT}/production.log"
printf 'complete\n' > "${EVIDENCE_ROOT}/status.txt"

archive="${WORKSPACE_ROOT}/detaudit-artifacts.tar.gz"
tar -czf "${archive}" -C "${WORKSPACE_ROOT}" \
  "$(basename "${OUTPUT_ROOT}")" \
  "$(basename "${PILOT_ROOT}")" \
  "$(basename "${EVIDENCE_ROOT}")"
sha256sum "${archive}" | tee "${archive}.sha256"

echo "Production matrix completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Download both files before stopping the Pod:"
echo "  ${archive}"
echo "  ${archive}.sha256"
