# Evaluation taxonomy

The evaluation converts both predictions and ground truth to a deliberately
small shared label space. Classes outside it are ignored rather than treated as
false negatives. The project deliberately does **not** create a synthetic
`cyclist` label by pairing people and bicycles: those are separately annotated
instances in COCO and nuImages.

| Evaluation class | COCO category | nuImages category families |
|---|---|---|
| `person` | person | `human.pedestrian.adult`, `child`, `construction_worker`, `police_officer` |
| `bicycle` | bicycle | `vehicle.bicycle` |
| `motorcycle` | motorcycle | `vehicle.motorcycle` |
| `car` | car | `vehicle.car` |
| `bus` | bus | `vehicle.bus.bendy`, `vehicle.bus.rigid` |
| `truck` | truck | `vehicle.truck` |

nuImages categories without a clean COCO counterpart (for example trailers,
construction vehicles, barriers, and personal-mobility devices) are excluded.
The exact category-token mapping is serialized with every experiment.

This mapping is a protocol decision, not an assertion that all listed visual
categories are semantically identical. The final report will identify the exact
category IDs and the implementation function used to apply this mapping.

### Fairness rules

- All models use the same image set for a dataset and experiment seed.
- Model-native preprocessing is used and recorded. We do not force a shared
  resize policy that changes a checkpoint's intended evaluation recipe.
- Raw diagnostic predictions use a fixed low candidate score floor (`0.001`)
  and retain up to 300 detections per image. Primary COCO-style metrics use
  `maxDets=100`; these limits are reported separately.
- The NMS intervention changes only final-stage class-aware NMS IoU. Faster
  R-CNN RPN NMS and all other settings remain fixed.
- COCO metrics are computed from saved detections using the same evaluator and
  taxonomy mapping for every model.
- Fine-tuning test images are held out before any target-domain training set is
  created.
