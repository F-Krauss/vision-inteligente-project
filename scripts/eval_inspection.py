#!/usr/bin/env python3
"""Measure inspection accuracy against the labeled benchmark set.

"Accurate enough" is a number, not a vibe. This runs the deterministic
multi-annotated-reference consensus (`inspect_expected_pieces_against_references`)
over every labeled eval image in ``benchmarks/annotations/*.json`` and reports the
metrics that matter for an industrial pass/fail with a false-rejection priority:

  • false_approval_rate  — faulty mold predicted "correct" (the HARD GATE; target 0)
  • false_rejection_rate — good mold sent to review (operator load; lower is better)
  • auto_approval_rate    — good molds auto-approved without review
  • piece_recall          — of known-missing parts, fraction actually flagged

Each benchmark file already carries the ground truth:
  golden_images  → the annotated reference photos (per-part consensus references)
  pieces         → canonical part boxes (id + normalized bbox)
  eval_images    → [{path, missing_piece_ids}]; empty list = a true OK image

Usage:
  python3 -m scripts.eval_inspection                      # all benchmarks
  python3 -m scripts.eval_inspection --glob 'mold-a*'     # subset by slug
  python3 -m scripts.eval_inspection --max-false-approval 0.0 --report reports/eval.md

Exits non-zero when false_approval_rate exceeds --max-false-approval, so it can gate CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mold_inspection.piece_inspector import inspect_expected_pieces_against_references

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANNOTATIONS = REPO_ROOT / "benchmarks" / "annotations"


@dataclass
class Tally:
    total: int = 0
    ok_total: int = 0
    fault_total: int = 0
    false_approvals: int = 0          # fault → "correct" (unsafe)
    false_rejections: int = 0         # ok → "review"
    auto_approved: int = 0            # ok → "correct"
    pieces_missing_total: int = 0
    pieces_missing_flagged: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, other: "Tally") -> None:
        for name in (
            "total", "ok_total", "fault_total", "false_approvals", "false_rejections",
            "auto_approved", "pieces_missing_total", "pieces_missing_flagged",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.rows.extend(other.rows)

    @property
    def false_approval_rate(self) -> float:
        return self.false_approvals / self.fault_total if self.fault_total else 0.0

    @property
    def false_rejection_rate(self) -> float:
        return self.false_rejections / self.ok_total if self.ok_total else 0.0

    @property
    def auto_approval_rate(self) -> float:
        return self.auto_approved / self.ok_total if self.ok_total else 0.0

    @property
    def piece_recall(self) -> float:
        return self.pieces_missing_flagged / self.pieces_missing_total if self.pieces_missing_total else 1.0


def _references_from_benchmark(bench: dict[str, Any]) -> list[dict[str, Any]]:
    """Build annotated_references (image_path + canonical part boxes) from a benchmark."""
    boxes = [
        {
            "element_id": str(piece["id"]),
            "class_name": "control" if piece.get("control") else "piece",
            "bbox": [float(v) for v in piece["bbox_norm"]],
        }
        for piece in bench.get("pieces", [])
        if isinstance(piece.get("bbox_norm"), list) and len(piece["bbox_norm"]) == 4
    ]
    refs = []
    for golden in bench.get("golden_images", []):
        if Path(golden).exists():
            refs.append({"image_path": str(golden), "boxes": [dict(b) for b in boxes]})
    return refs


def _eval_one(bench: dict[str, Any], evidence_dir: Path | None) -> Tally:
    tally = Tally()
    slug = bench.get("slug", "unknown")
    references = _references_from_benchmark(bench)
    if not references:
        print(f"  [skip] {slug}: no readable golden images", file=sys.stderr)
        return tally

    for item in bench.get("eval_images", []):
        path = item.get("path")
        if not path or not Path(path).exists():
            continue
        truth_missing = {str(pid) for pid in item.get("missing_piece_ids", [])}
        is_fault = bool(truth_missing)

        result = inspect_expected_pieces_against_references(
            family=bench.get("dataset", "benchmark"),
            zone_id=slug,
            candidate_image_path=path,
            annotated_references=references,
            evidence_dir=evidence_dir,
        )
        if result is None:
            print(f"  [skip] {slug}: consensus returned None for {Path(path).name}", file=sys.stderr)
            continue

        status = result["status"]
        flagged = {
            str(f.get("piece_id")) for f in result.get("findings", []) if f.get("status") == "missing"
        }

        tally.total += 1
        if is_fault:
            tally.fault_total += 1
            tally.pieces_missing_total += len(truth_missing)
            tally.pieces_missing_flagged += len(truth_missing & flagged)
            if status == "correct":
                tally.false_approvals += 1
        else:
            tally.ok_total += 1
            if status == "correct":
                tally.auto_approved += 1
            else:
                tally.false_rejections += 1

        tally.rows.append(
            {
                "slug": slug,
                "image": Path(path).name,
                "truth": "fault" if is_fault else "ok",
                "predicted": status,
                "missing_truth": sorted(truth_missing),
                "missing_flagged": sorted(flagged),
                "outcome": _outcome(is_fault, status),
            }
        )
    return tally


def _outcome(is_fault: bool, status: str) -> str:
    if is_fault:
        return "FALSE_APPROVAL" if status == "correct" else "caught"
    return "false_rejection" if status == "review" else "auto_approved"


def _render_report(tally: Tally, max_false_approval: float) -> str:
    gate = "PASS" if tally.false_approval_rate <= max_false_approval else "FAIL"
    lines = [
        "# Inspection accuracy report",
        "",
        f"- Images evaluated: **{tally.total}** ({tally.ok_total} ok, {tally.fault_total} fault)",
        f"- **false_approval_rate: {tally.false_approval_rate:.3f}** "
        f"(gate ≤ {max_false_approval:.3f} → **{gate}**)",
        f"- false_rejection_rate: {tally.false_rejection_rate:.3f}",
        f"- auto_approval_rate: {tally.auto_approval_rate:.3f}",
        f"- piece_recall: {tally.piece_recall:.3f} "
        f"({tally.pieces_missing_flagged}/{tally.pieces_missing_total} missing parts flagged)",
        "",
        "| section | image | truth | predicted | outcome | missing(truth→flagged) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in tally.rows:
        lines.append(
            f"| {row['slug']} | {row['image']} | {row['truth']} | {row['predicted']} | "
            f"{row['outcome']} | {','.join(row['missing_truth']) or '—'} → "
            f"{','.join(row['missing_flagged']) or '—'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_inspection")
    parser.add_argument("--annotations-dir", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--glob", default="*.json", help="filename glob within annotations dir")
    parser.add_argument("--max-false-approval", type=float, default=0.0,
                        help="hard gate; exit non-zero if false_approval_rate exceeds this")
    parser.add_argument("--report", default=None, help="write the markdown report here")
    parser.add_argument("--evidence-dir", default=None, help="save consensus overlays here (slower)")
    args = parser.parse_args(argv)

    annotations_dir = Path(args.annotations_dir)
    files = sorted(annotations_dir.glob(args.glob))
    if not files:
        print(f"No benchmark files matched {annotations_dir}/{args.glob}", file=sys.stderr)
        return 2

    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else None
    overall = Tally()
    for path in files:
        try:
            bench = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  [skip] {path.name}: invalid JSON", file=sys.stderr)
            continue
        overall.add(_eval_one(bench, evidence_dir))

    report = _render_report(overall, args.max_false_approval)
    print(report)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        print(f"Wrote {report_path}", file=sys.stderr)

    if overall.fault_total and overall.false_approval_rate > args.max_false_approval:
        print(
            f"GATE FAILED: false_approval_rate {overall.false_approval_rate:.3f} "
            f"> {args.max_false_approval:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
