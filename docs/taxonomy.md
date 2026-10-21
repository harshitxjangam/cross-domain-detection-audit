# Evaluation taxonomy

The shared course storage does not contain nuImages, and course policy prohibits
private dataset copies. The evaluation therefore uses KITTI and nuScenes and
converts predictions and ground truth to a deliberately narrow shared label
space. Classes outside it are ignored rather than treated as false negatives.

| Evaluation class | COCO-pretrained detector label | KITTI label | nuScenes categories |
|---|---|---|---|
| `person` | person | `Pedestrian` | adult, child, construction worker, police officer pedestrian |
| `car` | car | `Car` | `vehicle.car` |
| `truck` | truck | `Truck` | `vehicle.truck` |

KITTI `Van`, `Cyclist`, `Person_sitting`, and `Tram`, and nuScenes categories
such as buses, bicycles, motorcycles, trailers, and barriers are excluded. This
avoids relabeling semantically different objects or constructing synthetic
classes. nuScenes boxes are explicitly documented as 3D annotations projected
into CAM_FRONT image coordinates; KITTI uses its native 2D boxes.

This mapping is a protocol decision, not an assertion that all listed visual
categories are semantically identical. The final report will identify the exact
category IDs and the implementation function used to apply this mapping.

### Fairness rules

- All models use the same declared validation images for a dataset.
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
