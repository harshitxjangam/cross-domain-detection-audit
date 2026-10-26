"""Minimal, auditable 2D readers for the evaluation datasets.

The primary protocol uses COCO val2017 and a documented KITTI holdout. The
optional nuScenes reader remains available for a later authorized extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .taxonomy import Driving3Taxonomy, kitti_to_id, nuscenes_to_id


@dataclass
class AuditSample:
    image_id: int
    image: Image.Image
    gt_boxes: np.ndarray
    gt_labels: np.ndarray


class CocoValDataset:
    """Read COCO val2017 with its native 2D boxes and a strict class filter."""

    def __init__(self, root: Path, taxonomy: Driving3Taxonomy, split: str = "val",
                 max_images: int | None = None):
        if split != "val":
            raise ValueError("COCO evaluation supports val2017 only")
        try:
            from pycocotools.coco import COCO
        except ImportError as error:
            raise RuntimeError("pycocotools is required for COCO evaluation") from error
        self.root, self.taxonomy = Path(root), taxonomy
        annotation_path = self.root / "annotations" / "instances_val2017.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Expected COCO val annotations: {annotation_path}")
        self.coco = COCO(str(annotation_path))
        self.ids = sorted(self.coco.getImgIds())
        if max_images:
            self.ids = self.ids[:max_images]
        if not self.ids:
            raise RuntimeError("COCO val2017 contains no images")
        categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_to_label = {
            int(category["id"]): label
            for category in categories
            if (label := taxonomy.name_to_id(category["name"])) is not None
        }

    @property
    def metadata(self) -> dict:
        return {"name": "COCO 2017", "split": "val2017",
                "annotation_type": "native human-labeled 2D boxes"}

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> AuditSample:
        image_id = int(self.ids[index])
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = self.root / "val2017" / image_info["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing COCO image: {image_path}")
        boxes, labels = [], []
        annotation_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        for annotation in self.coco.loadAnns(annotation_ids):
            label = self.category_to_label.get(int(annotation["category_id"]))
            if label is None or int(annotation.get("iscrowd", 0)):
                continue
            x, y, width, height = map(float, annotation["bbox"])
            if width > 0 and height > 0:
                boxes.append([x, y, x + width, y + height])
                labels.append(label)
        return AuditSample(image_id, Image.open(image_path).convert("RGB"), *_arrays(boxes, labels))


def _arrays(boxes: list[list[float]], labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(labels, dtype=np.int64).reshape(-1))


class Kitti2DDataset:
    """Read native KITTI 2D labels without modifying the dataset copy."""

    def __init__(self, root: Path, taxonomy: Driving3Taxonomy, split: str = "val",
                 max_images: int | None = None):
        if split not in {"train", "val"}:
            raise ValueError("KITTI evaluation supports the labeled train or val split only")
        self.root, self.taxonomy, self.split = Path(root), taxonomy, split
        split_file = self.root / "ImageSets" / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Expected declared KITTI split file: {split_file}")
        self.ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        if max_images:
            self.ids = self.ids[:max_images]
        if not self.ids:
            raise RuntimeError(f"KITTI split {split!r} contains no images")

    @property
    def metadata(self) -> dict:
        return {"name": "KITTI 2D object detection", "split": self.split,
                "annotation_type": "native human-labeled 2D boxes"}

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> AuditSample:
        sample_id = self.ids[index]
        image_path = self.root / "training" / "image_2" / f"{sample_id}.png"
        label_path = self.root / "training" / "label_2" / f"{sample_id}.txt"
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing KITTI image or label for id {sample_id}")
        boxes, labels = [], []
        for line in label_path.read_text().splitlines():
            fields = line.split()
            if len(fields) < 8:
                continue
            label = kitti_to_id(fields[0], self.taxonomy)
            if label is None:
                continue
            x1, y1, x2, y2 = map(float, fields[4:8])
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(label)
        return AuditSample(index, Image.open(image_path).convert("RGB"), *_arrays(boxes, labels))


class NuScenesFrontDataset:
    """Project official nuScenes validation annotations into CAM_FRONT 2D boxes."""

    def __init__(self, root: Path, taxonomy: Driving3Taxonomy, split: str = "val",
                 camera: str = "CAM_FRONT", max_images: int | None = None):
        if split != "val":
            raise ValueError("This audit is fixed to the official nuScenes validation scenes")
        if camera != "CAM_FRONT":
            raise ValueError("This audit is fixed to CAM_FRONT for an explicit cross-domain protocol")
        try:
            from nuscenes.nuscenes import NuScenes
            from nuscenes.utils.splits import create_splits_scenes
        except ImportError as error:
            raise RuntimeError("nuscenes-devkit is required; use the shared course environment") from error
        self.root, self.taxonomy, self.split, self.camera = Path(root), taxonomy, split, camera
        self.nusc = NuScenes(version="v1.0-trainval", dataroot=str(root), verbose=False)
        val_scenes = set(create_splits_scenes()["val"])
        scene_name = {scene["token"]: scene["name"] for scene in self.nusc.scene}
        self.records = sorted(
            (sample["timestamp"], sample["data"][camera])
            for sample in self.nusc.sample
            if scene_name[sample["scene_token"]] in val_scenes and camera in sample["data"]
        )
        if max_images:
            self.records = self.records[:max_images]
        if not self.records:
            raise RuntimeError("No official nuScenes validation CAM_FRONT samples found")

    @property
    def metadata(self) -> dict:
        return {"name": "nuScenes", "version": "v1.0-trainval", "split": "official val",
                "camera": self.camera, "annotation_type": "3D boxes projected to 2D camera boxes"}

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> AuditSample:
        from nuscenes.utils.geometry_utils import BoxVisibility, view_points

        _, sample_data_token = self.records[index]
        data_path, boxes_3d, intrinsic = self.nusc.get_sample_data(
            sample_data_token, box_vis_level=BoxVisibility.ANY)
        sample_data = self.nusc.get("sample_data", sample_data_token)
        width, height = sample_data["width"], sample_data["height"]
        boxes, labels = [], []
        for box in boxes_3d:
            label = nuscenes_to_id(box.name, self.taxonomy)
            if label is None:
                continue
            # Discard corners behind the camera before perspective division.
            # This retains partially visible objects while avoiding inverted
            # coordinates from negative depth.
            corners_3d = box.corners()
            in_front = corners_3d[2] > 1e-5
            if not np.any(in_front):
                continue
            corners = view_points(corners_3d[:, in_front], intrinsic, normalize=True)[:2]
            x1, y1 = np.maximum(corners.min(axis=1), 0)
            x2, y2 = np.minimum(corners.max(axis=1), [width, height])
            if x2 > x1 and y2 > y1:
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                labels.append(label)
        return AuditSample(index, Image.open(data_path).convert("RGB"), *_arrays(boxes, labels))


def build_dataset(name: str, root: Path, taxonomy: Driving3Taxonomy, *, split: str,
                  camera: str, max_images: int | None):
    if name == "coco":
        return CocoValDataset(root, taxonomy, split=split, max_images=max_images)
    if name == "kitti":
        return Kitti2DDataset(root, taxonomy, split=split, max_images=max_images)
    if name == "nuscenes":
        return NuScenesFrontDataset(root, taxonomy, split=split, camera=camera, max_images=max_images)
    raise ValueError(f"Unsupported audit dataset: {name}")
