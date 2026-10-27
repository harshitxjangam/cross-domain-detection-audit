#!/usr/bin/env bash
# Download the official KITTI object images and labels for the declared split.

set -euo pipefail

if [[ "${KITTI_TERMS_ACCEPTED:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
KITTI_TERMS_ACCEPTED is not set to 1.
Register for KITTI access and review the official dataset terms first:
https://www.cvlibs.net/datasets/kitti/user_register.php

Then run:
  KITTI_TERMS_ACCEPTED=1 bash scripts/cloud/download_kitti_object.sh
EOF
  exit 2
fi

DATA_ROOT="${1:-/workspace/detaudit-data/kitti}"
WORK_DIR="${DATA_ROOT}/.downloads"
IMAGE_URL="${KITTI_IMAGE_URL:-https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip}"
LABEL_URL="${KITTI_LABEL_URL:-https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip}"
mkdir -p "${WORK_DIR}" "${DATA_ROOT}"

fetch() {
  local url=$1
  local destination=$2
  if [[ -f "${destination}" ]] && unzip -tq "${destination}" >/dev/null 2>&1; then
    return
  fi
  curl --fail --location --retry 5 --retry-all-errors \
    --continue-at - --output "${destination}" "${url}"
  unzip -tq "${destination}" >/dev/null
}

fetch "${IMAGE_URL}" "${WORK_DIR}/data_object_image_2.zip"
fetch "${LABEL_URL}" "${WORK_DIR}/data_object_label_2.zip"

# The official image archive also contains unlabeled test images. Extract only
# the labeled training images used by this project.
unzip -q -n "${WORK_DIR}/data_object_image_2.zip" "training/image_2/*" -d "${DATA_ROOT}"
unzip -q -n "${WORK_DIR}/data_object_label_2.zip" "training/label_2/*" -d "${DATA_ROOT}"

image_count=$(find "${DATA_ROOT}/training/image_2" -type f -name '*.png' | wc -l | tr -d ' ')
label_count=$(find "${DATA_ROOT}/training/label_2" -type f -name '*.txt' | wc -l | tr -d ' ')
if [[ "${image_count}" != "7481" || "${label_count}" != "7481" ]]; then
  echo "Expected 7481 KITTI training images and labels; found ${image_count} and ${label_count}." >&2
  exit 1
fi

python "$(dirname "$0")/prepare_kitti_split.py" \
  --root "${DATA_ROOT}" --val-count 1500 --seed 2026

rm -f "${WORK_DIR}/data_object_image_2.zip" "${WORK_DIR}/data_object_label_2.zip"
rmdir "${WORK_DIR}" 2>/dev/null || true
echo "KITTI object training data and deterministic split ready at ${DATA_ROOT}"
