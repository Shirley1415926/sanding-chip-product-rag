"""Apply human decisions to an existing generation-evaluation run, fully offline.

This command intentionally imports neither LLM clients, embedding providers,
Chroma, runtime configuration nor dotenv helpers. It only updates one existing
local run's JSON/Markdown artifacts and never makes a network request.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sanding_rag.generation_evaluation import (  # noqa: E402
    apply_manual_review_plan,
    failure_examples,
    summarize,
)
from sanding_rag.generation_evaluation_artifacts import demo_gate, render_report  # noqa: E402


def _arguments() -> object:
    parser = ArgumentParser(description="Apply selected human reviews to one local generation-evaluation run")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "runtime" / "generation_evaluation" / "latest",
        help="directory containing GENERATION_EVALUATION_TRACES.json and .md report",
    )
    parser.add_argument(
        "--review-file",
        type=Path,
        required=True,
        help="local JSON mapping selected case IDs to {decision: pass|fail, notes}",
    )
    return parser.parse_args()


def _load_reviews(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "reviews" in raw:
        raw = raw["reviews"]
    if not isinstance(raw, dict):
        raise ValueError("manual review JSON must be an object keyed by case ID")
    reviews: dict[str, dict[str, Any]] = {}
    for case_id, review in raw.items():
        if not isinstance(review, dict):
            raise ValueError(f"manual review for {case_id} must be an object")
        reviews[str(case_id)] = dict(review)
    return reviews


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def apply_review_to_run(run_dir: Path, reviews: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pure function with local file I/O only; exposed for no-API integration tests."""
    traces_path = run_dir / "GENERATION_EVALUATION_TRACES.json"
    report_path = run_dir / "GENERATION_EVALUATION_REPORT.md"
    if not traces_path.exists():
        raise ValueError(f"generation evaluation trace does not exist: {traces_path}")
    report = json.loads(traces_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise ValueError("manual review can be applied only to a completed generation-evaluation run")
    traces = report.get("traces")
    if not isinstance(traces, list):
        raise ValueError("completed generation evaluation trace must contain a traces list")

    apply_manual_review_plan(traces, reviews)
    summary = summarize(traces)
    report["summary"] = summary
    report["failure_examples"] = failure_examples(traces)
    report["demo_gate"] = demo_gate(summary)
    report["manual_review_updated_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    traces_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(report), encoding="utf-8")
    return report


def main() -> int:
    args = _arguments()
    try:
        report = apply_review_to_run(_resolve(args.run_dir), _load_reviews(_resolve(args.review_file)))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = report["summary"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "manual_review": summary["manual_review"],
                "demo_gate": report["demo_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
