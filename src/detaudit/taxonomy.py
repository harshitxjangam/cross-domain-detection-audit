"""Strict six-class intersection for COCO and nuImages.

This intentionally does not construct a cyclist label from person/bicycle pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


CLASSES = ("person", "bicycle", "motorcycle", "car", "bus", "truck")

NUIMAGES_EXACT = {
    "vehicle.car": "car", "vehicle.bicycle": "bicycle",
    "vehicle.motorcycle": "motorcycle", "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus", "vehicle.bus.rigid": "bus",
    "human.pedestrian.adult": "person", "human.pedestrian.child": "person",
    "human.pedestrian.construction_worker": "person",
    "human.pedestrian.police_officer": "person",
}


@dataclass(frozen=True)
class SixClassTaxonomy:
    """Duck-types the course harness Taxonomy interface used by its adapters."""

    name: str = "coco_nuimages6"
    classes: tuple[str, ...] = CLASSES

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def name_to_id(self, name: str | None) -> Optional[int]:
        if name is None:
            return None
        token = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "person": "person", "pedestrian": "person", "adult": "person",
            "child": "person", "construction_worker": "person",
            "police_officer": "person", "bicycle": "bicycle", "bike": "bicycle",
            "motorcycle": "motorcycle", "motorbike": "motorcycle", "car": "car",
            "bus": "bus", "truck": "truck", "bendy": "bus", "rigid": "bus",
        }
        # The course nuImages adapter normally calls this one component at a
        # time. Supporting exact dotted names as well keeps the mapping usable
        # and testable outside that adapter, without admitting broad vehicles.
        unified = NUIMAGES_EXACT.get(token) or aliases.get(token)
        return self.classes.index(unified) if unified else None

    def build_id_lut(self, source_id2name: Mapping[int, str]) -> dict[int, Optional[int]]:
        return {int(source_id): self.name_to_id(name)
                for source_id, name in source_id2name.items()}


def nuimages_to_id(dotted_name: str, taxonomy: SixClassTaxonomy) -> Optional[int]:
    """Map exact nuImages families; reject categories with no clean COCO match."""
    unified = NUIMAGES_EXACT.get(str(dotted_name).lower())
    return taxonomy.name_to_id(unified)
