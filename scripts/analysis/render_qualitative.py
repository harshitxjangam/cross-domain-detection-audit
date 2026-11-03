#!/usr/bin/env python3
"""Render a saved detector record against its original public image.

This utility performs no inference. It draws ground truth and already saved
predictions, making qualitative figures traceable to the same records used by
the quantitative report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


NAMES = ("person", "car", "truck")
COLORS = {0: "#2878B5", 1: "#35A271", 2: "#D95F02"}


def draw_boxes(image: Image.Image, boxes: list, labels: list, scores: list | None = None) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (box, label) in enumerate(zip(boxes, labels)):
        color = COLORS[int(label)]
        x1, y1, x2, y2 = map(float, box)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = NAMES[int(label)]
        if scores is not None:
            text = f"{text} {float(scores[index]):.2f}"
        left, top, right, bottom = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle((x1, max(0, y1 - (bottom - top) - 3), x1 + (right - left) + 4, y1), fill=color)
        draw.text((x1 + 2, max(0, y1 - (bottom - top) - 2)), text, fill="white", font=font)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Render saved ground truth and predictions side by side.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--image-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=.25)
    parser.add_argument("--scale", type=int, default=1,
                        help="Integer output upscaling factor for presentation only.")
    args = parser.parse_args()
    if args.scale < 1:
        parser.error("--scale must be at least 1")

    records = json.loads(args.records.read_text(encoding="utf-8"))
    record = next((row for row in records if int(row["image_id"]) == args.image_id), None)
    if record is None:
        raise SystemExit(f"No saved record for image ID {args.image_id}")
    image = Image.open(args.image).convert("RGB")
    if image.size != (record["width"], record["height"]):
        raise SystemExit(f"Image size {image.size} does not match saved record dimensions")

    selected = [index for index, score in enumerate(record["pred_scores"]) if score >= args.min_score]
    prediction_boxes = [record["pred_boxes_xyxy"][index] for index in selected]
    prediction_labels = [record["pred_labels"][index] for index in selected]
    prediction_scores = [record["pred_scores"][index] for index in selected]
    left = draw_boxes(image, record["gt_boxes_xyxy"], record["gt_labels"])
    right = draw_boxes(image, prediction_boxes, prediction_labels, prediction_scores)

    width, height = image.size
    header = 38
    canvas = Image.new("RGB", (width * 2, height + header), "white")
    canvas.paste(left, (0, header))
    canvas.paste(right, (width, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 12), "Ground truth: person blue, car green, truck orange", fill="black", font=font)
    draw.text((width + 8, 12), f"YOLO11s predictions, score >= {args.min_score:.2f}", fill="black", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.scale > 1:
        canvas = canvas.resize((canvas.width * args.scale, canvas.height * args.scale), Image.Resampling.LANCZOS)
    canvas.save(args.output)


if __name__ == "__main__":
    main()
