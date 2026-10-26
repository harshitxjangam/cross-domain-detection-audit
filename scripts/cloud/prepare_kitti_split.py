#!/usr/bin/env python3
"""Create a deterministic held-out KITTI evaluation split in a writable copy."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def rank(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible KITTI train/val ID lists.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--val-count", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    image_dir = args.root / "training" / "image_2"
    label_dir = args.root / "training" / "label_2"
    ids = sorted(
        path.stem for path in image_dir.glob("*.png")
        if (label_dir / f"{path.stem}.txt").is_file()
    )
    if args.val_count <= 0 or args.val_count >= len(ids):
        raise SystemExit(f"--val-count must be between 1 and {len(ids) - 1}")
    val_ids = sorted(sorted(ids, key=lambda sample_id: rank(sample_id, args.seed))[:args.val_count])
    val_set = set(val_ids)
    train_ids = [sample_id for sample_id in ids if sample_id not in val_set]
    split_dir = args.root / "ImageSets"
    split_dir.mkdir(exist_ok=True)
    (split_dir / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    (split_dir / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    print({"root": str(args.root), "seed": args.seed, "train_images": len(train_ids),
           "val_images": len(val_ids), "val_file": str(split_dir / "val.txt")})


if __name__ == "__main__":
    main()
