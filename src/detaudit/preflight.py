"""CPU-only validation for the shared KITTI and nuScenes audit inputs."""

from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args()
    print(json.dumps({
        "kitti": describe("kitti", args.kitti_root),
        "nuscenes": describe("nuscenes", args.nuscenes_root),
    }, indent=2))


if __name__ == "__main__":
    main()
