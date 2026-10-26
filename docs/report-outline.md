# Report outline (target: 8 pages)

This is a writing checklist, not a pre-written report. Fill every quantitative
claim from the saved artifacts produced by the commands in `scripts/hpc/`.

## 1. Motivation and question (0.5 page)

State the COCO-pretrained, zero-shot transfer question from COCO to KITTI and
the specific hypothesis:
final-stage duplicate suppression may explain some dense-detector failures, but
cannot explain misses caused by classification, localization, or recall.
Explicitly state that this is an empirical audit, not a new detector or a SOTA
claim.

## 2. Protocol and reproducibility (1--1.5 pages)

Describe the four checkpoints, the source and target validation domains, strict three-class
mapping, model-native preprocessing, score floor (0.001), saved-candidate cap
(300), metric cap (100), hardware, versions, and Slurm job IDs. Reference a
table populated directly from `metrics.json`.

## 3. Baseline cross-domain results (1.5--2 pages)

Report mAP@[.50:.95], AP50, AP75, AP small/medium/large, AR, and per-class AP
on COCO and KITTI. Make the cross-domain conclusion conditional on the actual
numbers. Include the first technical analysis: which class/scale degrades most?

## 4. Architecture and efficiency comparison (1--1.5 pages)

Contrast one-stage NMS-based YOLO, two-stage Faster R-CNN, and query-based,
NMS-free RT-DETR. Plot target-domain accuracy against measured median latency
and peak CUDA memory on the same GPU type. Do not compare latency across
different GPU models as if it were a model effect.

## 5. Failure anatomy and NMS intervention (1.5--2 pages)

Show TIDE dAP errors (`Cls`, `Loc`, `Both`, `Dupe`, `Bkg`, `Miss`) and the
original-coordinate scale plus same-class/all-class density strata. Then show
the controlled YOLO/Faster R-CNN final-stage NMS-IoU sweep. Explain that TIDE
`Dupe` is observational; the fixed-weight threshold sweep is the causal test.
State that Faster R-CNN RPN NMS was fixed.

## 6. Qualitative failures, limitations, conclusion (0.75--1 page)

Include a small number of linked failure crops selected from saved records,
with no cherry-picking claim. Discuss taxonomy filtering, checkpoint-native
resize differences, zero-shot-only scope, and any inconclusive/negative NMS
result. End with the concrete supported conclusion.

## References

Use `docs/references.bib` as the starter list and add the exact dataset/API
version pages used during execution. The report needs at least eight credible
technical references, including four original papers and two official
implementation/documentation sources.
