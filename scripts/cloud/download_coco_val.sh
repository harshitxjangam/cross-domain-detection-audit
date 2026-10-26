#!/usr/bin/env bash
# Download only the official COCO val2017 images and annotations needed here.
# This script downloads data into the active Pod workspace and never touches Git.

set -euo pipefail

DATA_ROOT="${1:-/workspace/detaudit-data/coco}"
WORK_DIR="${DATA_ROOT}/.downloads"
mkdir -p "${WORK_DIR}" "${DATA_ROOT}"

fetch() {
  local url=$1
  local destination=$2
  if [[ ! -f "${destination}" ]]; then
    curl --fail --location --retry 3 --output "${destination}" "${url}"
  fi
}

fetch "https://images.cocodataset.org/zips/val2017.zip" "${WORK_DIR}/val2017.zip"
fetch "https://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
  "${WORK_DIR}/annotations_trainval2017.zip"

unzip -q -n "${WORK_DIR}/val2017.zip" -d "${DATA_ROOT}"
unzip -q -n "${WORK_DIR}/annotations_trainval2017.zip" \
  "annotations/instances_val2017.json" -d "${DATA_ROOT}"

rm -f "${WORK_DIR}/val2017.zip" "${WORK_DIR}/annotations_trainval2017.zip"
rmdir "${WORK_DIR}" 2>/dev/null || true

test -f "${DATA_ROOT}/annotations/instances_val2017.json"
test -d "${DATA_ROOT}/val2017"
echo "COCO val2017 ready at ${DATA_ROOT}"
