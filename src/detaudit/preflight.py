"""CPU-only validation for the shared KITTI and nuScenes audit inputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from .datasets import build_dataset
from .taxonomy import Driving3Taxonomy


def describe(dataset_name: str, root: Path) -> dict:
    taxonomy = Driving3Taxonomy()
    dataset = build_dataset(dataset_name, root, taxonomy, split="val", camera="CAM_FRONT", max_images=1)
    sample = dataset[0]
    return {
        "dataset": dataset_name,
        "root": str(root),
        "metadata": dataset.metadata,
        "checked_images": len(dataset),
        "first_image_size": list(sample.image.size),
        "first_gt_boxes": int(len(sample.gt_boxes)),
        "first_gt_labels": sample.gt_labels.tolist(),
        "taxonomy": list(taxonomy.classes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate audit datasets without GPU inference.")
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream_root))
    from DeepDataMiningLearning.ngdet.detectors.base import build_detector

    required_packages = ("pycocotools", "tidecv", "ultralytics", "transformers")
    missing_packages = [name for name in required_packages if importlib.util.find_spec(name) is None]
    if missing_packages:
        raise RuntimeError(
            "Missing runtime packages: " + ", ".join(missing_packages) +
            ". Install them in the personal virtual environment before requesting a GPU.")
    print(json.dumps({
        "upstream_build_detector": str(build_detector),
        "runtime_packages": list(required_packages),
        "kitti": describe("kitti", args.kitti_root),
        "nuscenes": describe("nuscenes", args.nuscenes_root),
    }, indent=2))


if __name__ == "__main__":
    main()
