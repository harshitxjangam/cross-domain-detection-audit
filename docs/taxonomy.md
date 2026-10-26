# Evaluation taxonomy

The experiment evaluates COCO val2017 as the source domain and KITTI as the
autonomous-driving target domain. Both datasets are converted to a deliberately
narrow shared label space. Classes outside it are ignored rather than treated
as false negatives.

| Evaluation class | COCO ground truth and detector label | KITTI label |
|---|---|---|
| `person` | person | `Pedestrian` |
| `car` | car | `Car` |
| `truck` | truck | `Truck` |

KITTI `Van`, `Cyclist`, `Person_sitting`, and `Tram`, plus every COCO category
outside the three target classes, are excluded. This avoids relabeling
semantically different objects or constructing synthetic classes. Both selected
datasets use native human-labeled 2D boxes.

This mapping is a protocol decision, not an assertion that all listed visual
categories are semantically identical. The final report will identify the exact
category IDs and the implementation function used to apply this mapping.

### Fairness rules

- All models use the same declared image manifest for a dataset. COCO uses
  val2017; KITTI uses the repository's deterministic held-out split.
- Model-native preprocessing is used and recorded. We do not force a shared
  resize policy that changes a checkpoint's intended evaluation recipe.
- Raw diagnostic predictions use a fixed low candidate score floor (`0.001`)
  and retain up to 300 detections per image. Primary COCO-style metrics use
  `maxDets=100`; these limits are reported separately.
- The NMS intervention changes only final-stage class-aware NMS IoU. Faster
  R-CNN RPN NMS and all other settings remain fixed.
- COCO-style metrics are computed from saved detections using the same
  evaluator and taxonomy mapping for every model. They are three-class audit
  metrics, not the datasets' official full-category leaderboard scores.
- Fine-tuning test images are held out before any target-domain training set is
  created.
