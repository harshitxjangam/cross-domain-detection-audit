# Course starter repository

The project builds on the course `ngdet` harness from
[`lkk688/DeepDataMiningLearning`](https://github.com/lkk688/DeepDataMiningLearning).
It supplies detector adapters, dataset readers, taxonomy projection, COCO
evaluation, latency measurement, and report utilities.

## Integration rule

The upstream repository is a dependency, not copied submission material. Clone
it into the ignored `external/` directory locally or into persistent HPC project
storage, record its exact commit SHA in each completed experiment, and keep
project-specific code/configuration in this repository.

## Project-specific extension

This project adds a six-class COCO/nuImages protocol, deterministic manifests,
NMS-IoU intervention experiments, object-density/overlap strata, and TIDE error
analysis. These components are intentionally separated from upstream so a
grader can see exactly what was added.
