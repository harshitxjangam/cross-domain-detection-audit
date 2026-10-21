"""Strict three-class intersection for KITTI and nuScenes camera evaluation.

The course cluster does not host nuImages. This project therefore evaluates
COCO-pretrained detectors on two available autonomous-driving domains, using
only object categories with unambiguous equivalents in both datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


CLASSES = ("person", "car", "truck")

# Exact labels admitted from each ground-truth dataset. In particular, KITTI
# Van and Cyclist, and nuScenes bicycle/motorcycle/bus/trailer, are excluded:
# folding any of them into another label would contaminate the error audit.
KITTI_EXACT = {"car": "car", "truck": "truck", "pedestrian": "person"}
NUSCENES_EXACT = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "human.pedestrian.adult": "person",
    "human.pedestrian.child": "person",
    "human.pedestrian.construction_worker": "person",
    "human.pedestrian.police_officer": "person",
}


@dataclass(frozen=True)
class Driving3Taxonomy:
    """Duck-types the taxonomy interface used by the course detector adapters."""

    name: str = "kitti_nuscenes3"
    classes: tuple[str, ...] = CLASSES

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def name_to_id(self, name: str | None) -> Optional[int]:
        if name is None:
            return None
        token = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            # COCO checkpoint class names emitted by every selected detector.
            "person": "person", "car": "car", "truck": "truck",
            # Dataset labels are restricted by the exact maps above, but these
            # aliases let the upstream name-based detector folding work.
            "pedestrian": "person",
        }
        unified = NUSCENES_EXACT.get(token) or KITTI_EXACT.get(token) or aliases.get(token)
        return self.classes.index(unified) if unified else None

    def build_id_lut(self, source_id2name: Mapping[int, str]) -> dict[int, Optional[int]]:
        return {int(source_id): self.name_to_id(name)
                for source_id, name in source_id2name.items()}


def kitti_to_id(name: str, taxonomy: Driving3Taxonomy) -> Optional[int]:
    return taxonomy.name_to_id(KITTI_EXACT.get(str(name).lower()))


def nuscenes_to_id(name: str, taxonomy: Driving3Taxonomy) -> Optional[int]:
    return taxonomy.name_to_id(NUSCENES_EXACT.get(str(name).lower()))
