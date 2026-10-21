"""Small CPU-only checks for portable evaluation artifacts.

These do not test GPU model inference. They ensure that the analysis conducted
after an HPC run is deterministic and independent of the detector framework.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from detaudit.analyze import strata
from detaudit.datasets import Kitti2DDataset
from detaudit.run import coco_metrics, configure_detector
from detaudit.taxonomy import Driving3Taxonomy, kitti_to_id, nuscenes_to_id
from detaudit.tide_report import main as tide_main


def record(*, duplicate: bool = False) -> dict:
    predictions = [[10, 10, 30, 30]]
    scores = [0.95]
    labels = [0]
    if duplicate:
        predictions.append([10, 10, 30, 30])
        scores.append(0.80)
        labels.append(0)
    return {
        "image_id": 1,
        "dataset": "synthetic",
        "width": 100,
        "height": 100,
        "gt_boxes_xyxy": [[10, 10, 30, 30], [50, 50, 70, 70]],
        "gt_labels": [0, 1],
        "pred_boxes_xyxy": predictions,
        "pred_scores": scores,
        "pred_labels": labels,
    }


class TestTaxonomyAndDatasets(unittest.TestCase):
    def test_strict_kitti_nuscenes_intersection(self):
        taxonomy = Driving3Taxonomy()
        self.assertEqual(kitti_to_id("Pedestrian", taxonomy), 0)
        self.assertEqual(nuscenes_to_id("vehicle.truck", taxonomy), 2)
        self.assertIsNone(kitti_to_id("Van", taxonomy))
        self.assertIsNone(nuscenes_to_id("vehicle.trailer", taxonomy))

    def test_kitti_reader_uses_declared_split_and_drops_unmapped_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ImageSets").mkdir()
            (root / "training" / "image_2").mkdir(parents=True)
            (root / "training" / "label_2").mkdir(parents=True)
            (root / "ImageSets" / "val.txt").write_text("000001\n", encoding="utf-8")
            Image.new("RGB", (100, 80)).save(root / "training" / "image_2" / "000001.png")
            (root / "training" / "label_2" / "000001.txt").write_text(
                "Car 0 0 0 1 2 30 40 0 0 0 0 0 0 0\n"
                "Van 0 0 0 1 2 30 40 0 0 0 0 0 0 0\n", encoding="utf-8")
            dataset = Kitti2DDataset(root, Driving3Taxonomy(), split="val")
            sample = dataset[0]
            self.assertEqual(sample.image.size, (100, 80))
            self.assertEqual(sample.gt_boxes.tolist(), [[1.0, 2.0, 30.0, 40.0]])
            self.assertEqual(sample.gt_labels.tolist(), [1])


class TestArtifactAnalysis(unittest.TestCase):
    def test_faster_rcnn_sweep_changes_only_final_roi_nms(self):
        roi_heads = SimpleNamespace(score_thresh=0.5, detections_per_img=100, nms_thresh=0.5)
        rpn = SimpleNamespace(nms_thresh=0.7)
        detector = SimpleNamespace(family="torchvision", score_thr=0.001,
                                   model=SimpleNamespace(roi_heads=roi_heads, rpn=rpn))
        _, postprocessing = configure_detector(detector, nms_iou=0.9, max_detections=300, yolo_imgsz=640)
        self.assertEqual(roi_heads.score_thresh, 0.001)
        self.assertEqual(roi_heads.detections_per_img, 300)
        self.assertEqual(roi_heads.nms_thresh, 0.9)
        self.assertEqual(rpn.nms_thresh, 0.7)
        self.assertEqual(postprocessing, {"final_nms_iou": 0.9, "nms_applied": True,
                                          "rpn_nms_changed": False})

    def test_strata_uses_original_area_and_separate_density_dimensions(self):
        summary = strata([record()])
        self.assertEqual(summary["scale/small"], 2)
        self.assertEqual(summary["same_class/same_isolated"], 2)
        self.assertEqual(summary["all_class/all_isolated"], 2)

    def test_coco_metrics_evaluate_shared_three_class_space(self):
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = coco_metrics([record()], Driving3Taxonomy())
        self.assertAlmostEqual(metrics["AP50"], 0.5, places=6)
        self.assertAlmostEqual(metrics["AR_100"], 0.5, places=6)
        self.assertEqual(set(metrics["per_class"]), set(Driving3Taxonomy().classes))

    def test_empty_predictions_keep_the_same_metric_schema(self):
        empty = record()
        empty.update({"pred_boxes_xyxy": [], "pred_scores": [], "pred_labels": []})
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = coco_metrics([empty], Driving3Taxonomy())
        self.assertEqual(metrics["AP_small"], 0.0)
        self.assertEqual(metrics["AR_100"], 0.0)
        self.assertEqual(len(metrics["coco_stats"]), 12)

    def test_tide_report_is_generated_from_saved_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records.json"
            records.write_text(json.dumps([record(duplicate=True)]), encoding="utf-8")
            out = root / "tide"
            with patch.object(sys, "argv", ["detaudit-tide", str(records), "--out-dir", str(out)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    tide_main()
            report = json.loads((out / "tide_report.json").read_text(encoding="utf-8"))
            self.assertIn("Dupe", report["main_error_dAP"]["tide_predictions"])
            self.assertEqual(report["source_records"], str(records))


if __name__ == "__main__":
    unittest.main()
