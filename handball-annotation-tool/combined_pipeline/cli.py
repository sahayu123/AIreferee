from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import ParallelRefereePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the independent handball GRU and main-branch general-foul "
            "model, then fuse only their final decisions."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--config",
        default="configs/parallel_pipeline.yaml",
        type=Path,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--incident-time",
        type=float,
        help=(
            "Center time in seconds for the shared 41-frame incident window; "
            "defaults to the middle of the uploaded clip."
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check models, weights, packages, and API credentials only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = ParallelRefereePipeline.from_config(args.config)
    if args.preflight:
        report = pipeline.preflight()
        print(json.dumps(report, indent=2))
        if not all(
            bool(section["available"])
            for section in report.values()
        ):
            raise SystemExit(2)
        return
    result = pipeline.run(
        args.input,
        output_dir=args.output_dir,
        incident_time_seconds=args.incident_time,
        progress=print,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
