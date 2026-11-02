#!/usr/bin/env python3
"""Create compact figures, result tables, and the final technical report.

The script intentionally consumes only locally saved experiment artifacts. It
does not download models, datasets, or reference benchmark values.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "artifacts" / "final_evidence"
SUMMARY = EVIDENCE / "summary.json"
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
RESULTS_DIR = ROOT / "results"
PDF_PATH = REPORT_DIR / "cross_domain_detection_audit.pdf"

DISPLAY = {
    "yolo:yolo11s.pt": "YOLO11s",
    "yolo:yolo11m.pt": "YOLO11m",
    "torchvision:fasterrcnn_resnet50_fpn_v2": "Faster R-CNN R50-FPN v2",
    "hf_detr:PekingU/rtdetr_r50vd": "RT-DETR R50",
}
COLORS = {"YOLO11s": "#3566a8", "YOLO11m": "#7191c8", "Faster R-CNN R50-FPN v2": "#d1663d", "RT-DETR R50": "#3c9b77"}
TIDE_KEYS = ["TIDE_Cls", "TIDE_Loc", "TIDE_Both", "TIDE_Dupe", "TIDE_Bkg", "TIDE_Miss"]
TIDE_LABELS = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]


def load_rows() -> list[dict]:
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Missing saved evidence: {SUMMARY}")
    rows = json.loads(SUMMARY.read_text())
    if len(rows) != 18:
        raise ValueError(f"Expected 18 completed runs, found {len(rows)}")
    return rows


def label(row: dict) -> str:
    return DISPLAY[row["model"]]


def baseline_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["run"].endswith("1000")]


def setup_plot() -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9, "legend.fontsize": 8})


def save_domain_shift(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 3.25))
    models = ["YOLO11s", "YOLO11m", "Faster R-CNN R50-FPN v2", "RT-DETR R50"]
    base_rows = baseline_rows(rows)
    source = {label(r): r for r in base_rows if r["dataset"] == "coco"}
    target = {label(r): r for r in base_rows if r["dataset"] == "kitti"}
    x = list(range(len(models)))
    w = 0.36
    ax.bar([v - w / 2 for v in x], [source[m]["mAP"] for m in models], w, label="COCO val2017", color="#5e80b7")
    ax.bar([v + w / 2 for v in x], [target[m]["mAP"] for m in models], w, label="KITTI target", color="#e08a59")
    for i, m in enumerate(models):
        ax.text(i + w / 2, target[m]["mAP"] + .012, f"{target[m]['mAP']:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["YOLO11s", "YOLO11m", "Faster R-CNN", "RT-DETR"], rotation=0)
    ax.set_ylim(0, .55)
    ax.set_ylabel("mAP@[.50:.95]")
    ax.set_title("Zero-shot domain transfer on the shared three-class taxonomy")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    path = FIG_DIR / "domain_shift.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_pareto(rows: list[dict]) -> Path:
    target = [r for r in rows if r["dataset"] == "kitti"]
    fig, ax = plt.subplots(figsize=(6.9, 3.35))
    for r in target:
        name = label(r)
        ax.scatter(r["median_latency_ms"], r["mAP"], s=72, color=COLORS[name], zorder=3)
        ax.annotate(name.replace(" R50-FPN v2", ""), (r["median_latency_ms"], r["mAP"]), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Median end-to-end latency (ms/image, RTX A6000)")
    ax.set_ylabel("KITTI mAP@[.50:.95]")
    ax.set_title("Accuracy-latency comparison: baseline runs only")
    ax.grid(alpha=.25, which="both")
    fig.tight_layout()
    path = FIG_DIR / "accuracy_latency.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_nms(rows: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.1))
    families = [
        ("YOLO11s", [r for r in rows if r["run"].startswith("yolo11s-kitti-nms")]),
        ("Faster R-CNN", [r for r in rows if r["run"].startswith("fasterrcnn-kitti-nms")]),
    ]
    for family, group in families:
        group.sort(key=lambda r: r["nms_iou"])
        color = COLORS["YOLO11s"] if family == "YOLO11s" else COLORS["Faster R-CNN R50-FPN v2"]
        xs = [r["nms_iou"] for r in group]
        axes[0].plot(xs, [r["mAP"] for r in group], marker="o", label=family, color=color)
        axes[1].plot(xs, [r["TIDE_Dupe"] for r in group], marker="o", label=family, color=color)
    axes[0].set_ylabel("KITTI mAP@[.50:.95]")
    axes[1].set_ylabel("TIDE duplicate dAP")
    for ax, title in zip(axes, ["Accuracy response", "Surviving duplicate errors"]):
        ax.set_xlabel("Final detection NMS IoU threshold")
        ax.set_title(title)
        ax.set_xticks([.30, .45, .60, .75, .90])
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    fig.suptitle("Fixed-weight final-stage NMS intervention on KITTI", y=1.02, fontsize=11)
    fig.tight_layout()
    path = FIG_DIR / "nms_intervention.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_tide(rows: list[dict]) -> Path:
    target = [r for r in rows if r["dataset"] == "kitti"]
    names = [label(r) for r in target]
    fig, ax = plt.subplots(figsize=(7.0, 3.35))
    bottom = [0.0] * len(target)
    palette = ["#5875a4", "#cc8963", "#9dbf9e", "#c6b0d5", "#d1b561", "#83b4bd"]
    for key, key_label, color in zip(TIDE_KEYS, TIDE_LABELS, palette):
        values = [r[key] for r in target]
        ax.bar(range(len(target)), values, bottom=bottom, label=key_label, color=color, width=.7)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_xticks(range(len(target)), [n.replace(" R50-FPN v2", "") for n in names], rotation=0)
    ax.set_ylabel("TIDE dAP contribution (mAP percentage points)")
    ax.set_title("Target-domain TIDE error decomposition", pad=8)
    ax.legend(frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(.5, -.16))
    ax.grid(axis="y", alpha=.2)
    fig.subplots_adjust(bottom=.26, top=.88, left=.12, right=.98)
    path = FIG_DIR / "tide_target.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def copy_compact_results(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    keep = [
        "run", "model", "dataset", "nms_iou", "num_images", "mAP", "AP50", "AP75",
        "AP_small", "AP_medium", "AP_large", "AP_person", "AP_car", "AP_truck", "AR_100",
        "median_latency_ms", "peak_cuda_memory_mib", *TIDE_KEYS, "records_sha256",
    ]
    with (RESULTS_DIR / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keep, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: r.get(key) for key in keep} for r in rows])
    (RESULTS_DIR / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def tab(rows: list[list[str]], widths: list[float], repeat: int = 1) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=repeat, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#23415e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("LEADING", (0, 0), (-1, -1), 8.4),
        ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#c8d1da")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def make_report(rows: list[dict], figures: list[Path]) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=letter, rightMargin=.58 * inch, leftMargin=.58 * inch,
        topMargin=.53 * inch, bottomMargin=.50 * inch,
        title="Cross-Domain 2D Object Detection: A Post-Processing Failure Audit",
        author="Harshit Jangam",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=17, leading=20, textColor=colors.HexColor("#173a5e"), spaceAfter=5))
    styles.add(ParagraphStyle(name="AuthorCenter", parent=styles["Normal"], alignment=TA_CENTER, fontSize=9, leading=11, spaceAfter=11))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=12, leading=14, textColor=colors.HexColor("#173a5e"), spaceBefore=7, spaceAfter=4))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=10, leading=12, textColor=colors.HexColor("#23415e"), spaceBefore=5, spaceAfter=3))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=8.5, leading=10.7, spaceAfter=5))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.3, leading=8.5, textColor=colors.HexColor("#36454f"), spaceAfter=4))
    body, small, h1, h2 = styles["Bodyx"], styles["Smallx"], styles["H1x"], styles["H2x"]
    base_rows = baseline_rows(rows)
    source = {label(r): r for r in base_rows if r["dataset"] == "coco"}
    target = {label(r): r for r in base_rows if r["dataset"] == "kitti"}
    ordered = ["YOLO11s", "YOLO11m", "Faster R-CNN R50-FPN v2", "RT-DETR R50"]
    story = []
    story += [p("Cross-Domain 2D Object Detection: A Post-Processing Failure Audit", styles["TitleCenter"]),
              p("Harshit Jangam | CMPE 249 | Experimental Track", styles["AuthorCenter"])]
    story += [p("Abstract", h1), p(
        "This report evaluates four COCO-pretrained 2D object detector configurations without fine-tuning on a shared three-class protocol: person, car, and truck. The source domain is COCO val2017 and the target domain is KITTI. Rather than presenting one aggregate score, the study combines cross-domain metrics, scale and class breakdowns, latency and memory profiling, TIDE error decomposition, and a fixed-weight NMS-IoU intervention. All reported metrics come from deterministic 1,000-image saved prediction sets. The result is a controlled failure audit, not a claim of a new detector or state of the art.", body)]
    story += [p("1. Question and hypothesis", h1), p(
        "The primary question is: when a COCO-pretrained detector is transferred to an autonomous-driving domain without retraining, how much of the performance loss can be attributed to final duplicate suppression rather than upstream localization, background, classification, or recall failures? The testable hypothesis is that changing only final-stage NMS should affect duplicate errors and may recover some accuracy, but cannot remove upstream failure modes.", body)]
    story += [p("2. Experimental protocol", h1), p(
        "Four configurations span three detector families: YOLO11s and YOLO11m are one-stage dense-prediction models with NMS; Faster R-CNN R50-FPN v2 is a two-stage region-based detector with final RoI NMS; and RT-DETR R50 is a query-based, NMS-free detector. COCO val2017 and KITTI target images are filtered to person, car, and truck. COCO-only and KITTI-only labels are excluded rather than heuristically remapped. Each run evaluates a deterministic prefix of 1,000 images, uses a score floor of 0.001, saves at most 300 candidates, and evaluates at most 100 detections per image. Object-scale labels use original annotation coordinates, not post-resize coordinates.", body),
        p("Baseline measurements were run on a single RTX A6000 48 GB cloud GPU. The later NMS thresholds were executed on an RTX 4090 24 GB after recovery. Therefore, only baseline latency and memory values are used for cross-model efficiency comparison; NMS sweep latency is not compared across the two GPUs.", body)]
    protocol = [["Configuration", "Family", "Input preprocessing", "Baseline post-processing"]]
    protocol += [
        ["YOLO11s", "one-stage dense", "letterbox 640 x 640", "final class-aware NMS IoU 0.70"],
        ["YOLO11m", "one-stage dense", "letterbox 640 x 640", "final class-aware NMS IoU 0.70"],
        ["Faster R-CNN R50-FPN v2", "two-stage", "min side 800; max side 1333", "RoI NMS IoU 0.50; RPN fixed"],
        ["RT-DETR R50", "query-based", "processor resize 640 x 640", "set prediction; no final NMS"],
    ]
    story += [tab(protocol, [1.35*inch, 1.08*inch, 1.62*inch, 2.25*inch]), Spacer(1, 6)]
    story += [p("KITTI uses a generated 1,500-image held-out split: the official labeled training IDs are hash-ranked with seed 2026, the selected IDs are sorted, and the first 1,000 are evaluated. This is a reproducible split generated by `scripts/cloud/prepare_kitti_split.py`, not a published KITTI benchmark split; it makes no sequence-independence claim. The reference image size is 1242 x 375. The Faster R-CNN transform therefore scales the image subject to the 800-pixel short-side and 1333-pixel long-side bounds, while the two YOLO configurations and RT-DETR receive 640 x 640 tensors. This is intentionally a checkpoint-native comparison, not a resolution-matched comparison. Score floor is 0.001 for all models; it preserves the low-score tail needed for the precision-recall curve. The baseline NMS settings are independently validated by the sweep: the baseline values land between their adjacent sweep values.", body)]
    story += [p("3. Baseline cross-domain evaluation", h1), Image(str(figures[0]), width=6.75*inch, height=3.13*inch), PageBreak(), p("3. Baseline cross-domain evaluation (continued)", h1), p(
        "Figure 1 shows a consistent transfer gap. The target-domain mAP decreases by 0.143 for YOLO11s, 0.153 for YOLO11m, 0.146 for Faster R-CNN, and 0.144 for RT-DETR. Faster R-CNN has the highest baseline KITTI mAP (0.315), narrowly ahead of RT-DETR (0.308) and YOLO11m (0.301). The result does not establish an architecture as universally better: it is bounded to this three-class, zero-shot, 1,000-image protocol.", body)]
    base_table = [["Model", "COCO mAP", "KITTI mAP", "AP50", "AP75", "AP S/M/L", "person/car/truck AP"]]
    for name in ordered:
        s, t = source[name], target[name]
        base_table.append([name.replace(" R50-FPN v2", ""), f"{s['mAP']:.3f}", f"{t['mAP']:.3f}", f"{t['AP50']:.3f}", f"{t['AP75']:.3f}", f"{t['AP_small']:.3f}/{t['AP_medium']:.3f}/{t['AP_large']:.3f}", f"{t['AP_person']:.3f}/{t['AP_car']:.3f}/{t['AP_truck']:.3f}"])
    story += [tab(base_table, [1.15*inch, .63*inch, .66*inch, .50*inch, .50*inch, 1.20*inch, 1.45*inch]), Spacer(1, 5), p(
        "Across all four target-domain models, cars are substantially easier than persons and trucks. The lowest target AP values occur for YOLO11s truck (0.107) and YOLO11s small objects (0.176). The scale pattern is also consistent: large objects are easier than medium, which are easier than small, except RT-DETR's target small-object score (0.217) is comparatively stronger than the other configurations. These observations satisfy the cross-dataset and scale-conditioned analyses without implying that the finite target subset estimates the complete KITTI benchmark.", body), p(
        "The course ngdet tutorial reports the same broad pattern on its separate 120-image driving3 evaluation: Faster R-CNN 0.279, RT-DETR 0.262, and YOLO11x 0.260 on KITTI [10]. The protocols differ in taxonomy, model sizes, image count, score floor, and GPU, so these values are not a numerical reproduction. However, both studies show cross-domain compression: the present COCO-to-KITTI mAP drops span only 0.143 to 0.153 across three detector families. This audit extends that teaching observation by testing whether final-stage NMS accounts for a material portion of the residual gap.", body)]
    story += [PageBreak(), p("4. Architecture and efficiency comparison", h1), Image(str(figures[1]), width=6.72*inch, height=3.26*inch), p(
        "The baseline A6000 timing study shows the expected practical trade-off. YOLO11s is fastest at 11.1 ms/image (about 90 FPS) and uses 59 MiB peak allocated CUDA memory, but has the lowest target mAP. Faster R-CNN is most accurate in this protocol but has 90.1 ms median latency and 439 MiB peak allocated memory. RT-DETR is NMS-free at inference and has 31.1 ms median latency with 298 MiB allocated memory. All timings include model preprocessing and post-processing in the same single-image evaluation harness; they are not vendor benchmark claims.", body)]
    efficiency = [["Model", "KITTI mAP", "AR@100", "Median ms", "FPS", "Peak MiB"]]
    for name in ordered:
        r = target[name]
        efficiency.append([name.replace(" R50-FPN v2", ""), f"{r['mAP']:.3f}", f"{r['AR_100']:.3f}", f"{r['median_latency_ms']:.1f}", f"{r['fps']:.1f}", f"{r['peak_cuda_memory_mib']:.0f}"])
    story += [tab(efficiency, [1.55*inch, .82*inch, .78*inch, .86*inch, .67*inch, .83*inch]), Spacer(1, 6), p(
        "The comparison also clarifies what is and is not held constant. Architecture-specific preprocessing is preserved because forcing a two-stage detector into a square letterbox would introduce a confound. Evaluation annotations, taxonomy, score floor, candidate cap, metric cap, image manifest, batch size, and baseline GPU type are held fixed. The efficiency analysis is therefore a practical system comparison, not a claim that all architectures receive identical internal feature resolution. Input resolution alone does not explain the small-object ordering: Faster R-CNN receives the largest native input yet has the lowest target AP-small (0.161), while RT-DETR at 640 x 640 has the highest (0.217). The resolution difference remains a limitation, but it is not sufficient to explain this scale-conditioned ordering.", body)]
    story += [PageBreak(), p("5. Failure anatomy", h1), Image(str(figures[3]), width=6.72*inch, height=3.22*inch), p(
        "TIDE separates error contributions after prediction. The dAP axis is measured in mAP percentage points. On KITTI, background and localization errors dominate the visible error anatomy, while duplicates are comparatively small at the baseline thresholds. This motivates the intervention instead of assuming that NMS is the root cause. TIDE duplicate dAP is observational: it measures errors that survived post-processing, not a direct causal estimate of NMS behavior.", body)]
    strata = [["Target-domain ground truth stratum", "Count", "Definition"],
              ["small / medium / large", "790 / 2,416 / 1,281", "COCO area bands in original image coordinates"],
              ["same-class isolated / near / overlap", "2,579 / 1,102 / 806", "max same-class IoU < .10 / [.10, .30) / >= .30"],
              ["all-class isolated / near / overlap", "2,472 / 1,187 / 828", "same bands using any-class spatial neighbor"],
              ["person / car / truck", "646 / 3,685 / 156", "strict shared taxonomy only"],]
    story += [tab(strata, [2.20*inch, 1.50*inch, 3.05*inch]), Spacer(1, 6), p(
        "The density strata distinguish same-class overlap from any-class neighborhood density. That distinction matters because class-aware NMS acts within a class, whereas cross-class proximity can still cause occlusion and feature ambiguity without being directly suppressed. The target set contains relatively few trucks, so truck conclusions should be interpreted as evidence from the declared fixed subset rather than a broad population estimate.", body), PageBreak(), p("6. NMS intervention", h1), Image(str(figures[2]), width=6.72*inch, height=2.96*inch), p(
        "The causal test holds model weights, images, score floor, and saved-candidate cap fixed while sweeping only final detection NMS IoU. The largest observed point estimates occur at 0.75: 0.267 for YOLO11s and 0.317 for Faster R-CNN. However, the differences from 0.60 are only 0.0024 and 0.0007, respectively, and this study does not estimate confidence intervals. The correct interpretation is a flat intermediate-threshold plateau, not a resolved optimum. Both models fall at 0.90 while duplicate dAP rises sharply. For Faster R-CNN, RPN proposal NMS remained fixed, isolating final RoI suppression. Thus a looser threshold can recover some overlapping detections, but excessive permissiveness increases surviving duplicates and does not solve background or localization errors.", body)]
    sweep = [["Model", "NMS 0.30", "0.45", "0.60", "0.75", "0.90"]]
    for prefix, name in [("yolo11s-kitti-nms", "YOLO11s"), ("fasterrcnn-kitti-nms", "Faster R-CNN")]:
        group = sorted([r for r in rows if r["run"].startswith(prefix)], key=lambda x: x["nms_iou"])
        sweep.append([name] + [f"{r['mAP']:.3f}" for r in group])
    story += [tab(sweep, [1.35*inch] + [.84*inch]*5), Spacer(1, 6)]
    story += [PageBreak(), p("7. Limitations and conclusion", h1), p(
        "This is a zero-shot study, not a fine-tuning study. It uses a strict shared taxonomy, so non-target classes are excluded from the three-class evaluator. Excluded KITTI labels are removed from ground truth rather than retained as ignore regions; a prediction overlapping a removed object can therefore count as a background false positive. This is a deliberate closed-set choice because mapping a `Van` to `car` or `truck` would itself be a heuristic, but it means the TIDE background component is an upper bound rather than a clean estimate of detector hallucination. `Cyclist` and `Person_sitting` are important candidates for a future ignore-region sensitivity analysis. COCO crowd annotations in target classes were also dropped rather than retained as COCO ignore regions; accordingly, results are strict three-class audit metrics and are not comparable to official full-category COCO AP. A sensitivity re-audit with retained ignore regions would require the original annotation files because discarded boxes are not present in the saved records.", body), p(
        "Native preprocessing differs by model family and is documented rather than artificially forced to a single resize recipe; therefore the accuracy-latency comparison is checkpoint-native, not resolution-matched. Unlike the official KITTI benchmark, this audit scores every mapped object regardless of truncation, occlusion, or box height, so small-object results are intentionally not comparable to KITTI leaderboard difficulty tiers. The truck class has only 156 target instances and COCO/KITTI truck definitions do not align perfectly. The deterministic 1,000-image subsets support reproducibility and matched comparisons, but are not official full-dataset leaderboard evaluations. Finally, the NMS sweep and baseline speed study use different GPU models, so the report keeps their interpretations separate.", body), p(
        "The main conclusion is that the cross-domain drop is not reducible to duplicate suppression. Across the measured threshold range, YOLO11s gains 0.77 mAP points and Faster R-CNN gains 0.64 points from NMS 0.30 to 0.75, versus COCO-to-KITTI drops of 14.29 and 14.55 points. This bounds the observed final-stage NMS effect to about five percent of the domain gap in this protocol. The stronger target-domain models still exhibit substantial background and localization contributions. A useful next experiment would fine-tune one configuration on the target training split and repeat the same TIDE and NMS protocol with bootstrap confidence intervals and explicit ignore-region sensitivity analysis.", body)]
    story += [p("Evidence and reproducibility", h1), p(
        "The submitted repository includes versioned configurations, exact cloud commands, evaluation code, compact summary tables, and figure-generation code. Raw predictions and images are intentionally excluded from Git because they are large; each saved run has a SHA-256 hash recorded in the compact summary. The report is generated by scripts/analysis/build_submission.py directly from those saved artifacts.", body)]
    story += [p("References", h1)]
    refs = [
        "[1] S. Ren et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS, 2015.",
        "[2] N. Carion et al. End-to-End Object Detection with Transformers. ECCV, 2020.",
        "[3] Y. Zhao et al. DETRs Beat YOLOs on Real-time Object Detection. CVPR, 2024.",
        "[4] D. Bolya et al. TIDE: A General Toolbox for Identifying Object Detection Errors. ECCV, 2020.",
        "[5] T.-Y. Lin et al. Microsoft COCO: Common Objects in Context. ECCV, 2014.",
        "[6] A. Geiger et al. Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite. CVPR, 2012.",
        "[7] Ultralytics. YOLO Documentation. Official documentation, https://docs.ultralytics.com/.",
        "[8] PyTorch Contributors. Torchvision Object Detection Models and Weights. Official documentation, https://pytorch.org/vision/stable/models.html.",
        "[9] Hugging Face. Object Detection Task Guide. Official documentation, https://huggingface.co/docs/transformers/tasks/object_detection.",
        "[10] lkk688. DeepDataMiningLearning ngdet Tutorial. Course starter repository, https://github.com/lkk688/DeepDataMiningLearning.",
    ]
    story += [p(ref, small) for ref in refs]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#607080"))
        canvas.drawString(.58 * inch, .32 * inch, "Cross-Domain 2D Object Detection: A Post-Processing Failure Audit")
        canvas.drawRightString(7.92 * inch, .32 * inch, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    setup_plot()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    baseline = baseline_rows(rows)
    if len(baseline) != 8:
        raise ValueError(f"Expected 8 baseline rows, found {len(baseline)}")
    copy_compact_results(rows)
    figures = [save_domain_shift(baseline), save_pareto(baseline), save_nms(rows), save_tide(baseline)]
    make_report(rows, figures)
    print(f"Wrote compact results to {RESULTS_DIR}")
    print(f"Wrote report to {PDF_PATH}")


if __name__ == "__main__":
    main()
