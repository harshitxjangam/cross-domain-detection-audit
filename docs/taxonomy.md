# Evaluation taxonomy

The evaluation converts both predictions and ground truth to a deliberately
small common label space. Classes outside it are ignored rather than treated as
false negatives.

| Unified class | COCO categories | nuImages category families |
|---|---|---|
| `vehicle` | car, truck, bus | car, truck, bus, trailer, construction vehicle, emergency vehicle |
| `person` | person | pedestrian, wheelchair, stroller, personal mobility |
| `cyclist` | bicycle, motorcycle | bicycle, motorcycle |

This mapping is a protocol decision, not an assertion that all listed visual
categories are semantically identical. The final report will identify the exact
category IDs and the implementation function used to apply this mapping.

### Fairness rules

- All models use the same image set for a dataset and experiment seed.
- Evaluation resolution, maximum detections, score threshold, and NMS settings
  are recorded per model. A threshold sweep may be reported separately; it must
  not be confused with the primary fixed-threshold comparison.
- COCO metrics are computed from saved detections using the same evaluator and
  taxonomy mapping for every model.
- Fine-tuning test images are held out before any target-domain training set is
  created.
