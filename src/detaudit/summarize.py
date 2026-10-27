"""Build compact report tables from completed experiment directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .records import write_json


METRIC_FIELDS = (
    "mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_100", "AR_small", "AR_medium", "AR_large",
)
TIDE_FIELDS = ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_run(run_dir: Path) -> dict[str, Any]:
    payload = _read_json(run_dir / "metrics.json")
    metrics = payload["metrics"]
    metadata = payload["metadata"]
    row: dict[str, Any] = {
        "run": run_dir.name,
        "model": metadata["model"],
        "dataset": metadata["dataset"],
        "num_images": metadata["num_images"],
        "nms_iou": metadata.get("postprocessing", {}).get("final_nms_iou"),
        "mean_latency_ms": metadata.get("mean_latency_ms"),
        "median_latency_ms": metadata.get("median_latency_ms"),
        "fps": metadata.get("fps"),
        "peak_cuda_memory_mib": (
            metadata["peak_cuda_memory_bytes"] / 1024 ** 2
            if metadata.get("peak_cuda_memory_bytes") is not None else None
        ),
        "records_sha256": metadata.get("records_sha256"),
    }
    row.update({field: metrics.get(field) for field in METRIC_FIELDS})
    row.update({f"AP_{name}": value for name, value in metrics.get("per_class", {}).items()})

    tide_path = run_dir / "tide" / "tide_report.json"
    if tide_path.is_file():
        report = _read_json(tide_path)
        errors = report.get("main_error_dAP", {}).get("tide_predictions", {})
        row.update({f"TIDE_{field}": errors.get(field) for field in TIDE_FIELDS})
    return row


def discover_runs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.glob("*/metrics.json"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed detector audit runs.")
    parser.add_argument("root", type=Path, help="Directory containing one subdirectory per run")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    runs = discover_runs(args.root)
    if not runs:
        raise SystemExit(f"No */metrics.json files found under {args.root}")
    rows = [summarize_run(run) for run in runs]
    write_csv(args.csv, rows)
    write_json(args.json, rows)
    print({"runs": len(rows), "csv": str(args.csv), "json": str(args.json)})


if __name__ == "__main__":
    main()
