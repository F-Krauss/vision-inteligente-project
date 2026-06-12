"""Signal fusion, per-piece decisions, and multi-frame voting.

Scores are computed as deviations from MULTI-GOLDEN-FRAME statistics rather
than absolute thresholds: registration captures several golden frames, giving
each piece a per-signal mean/std under benign pose+lighting jitter. That is
what makes "embedding cosine 0.83" interpretable for a tiny screw and a large
plate alike.

Decision policy encodes prefer-false-rejection-over-false-approval:
- thresholds are calibrated so no labeled-missing sample scores above
  ``t_present`` (Phase 0 / recalibration fit),
- vote() auto-approves only with majority present votes AND zero missing votes.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal

# Directional signals: BELOW golden mean is evidence of change (higher = more similar).
_DIRECTIONAL = ("emb_cos", "interior_ncc", "grad_cos")
# Symmetric signals: any deviation from golden mean is evidence of change.
_SYMMETRIC = ("ring_ratio", "edge_density")

# Std floors: golden frames are near-identical, so raw stds collapse and tiny
# benign fluctuations would otherwise explode the z-scores.
_STD_FLOORS = {
    "emb_cos": 0.02,
    "interior_ncc": 0.08,
    "ring_ratio": 0.05,
    "edge_density": 0.03,
    "grad_cos": 0.03,
}

DEFAULT_WEIGHTS = {
    "emb_cos": 0.45,
    "interior_ncc": 0.20,
    "ring_ratio": 0.20,
    "edge_density": 0.075,
    "grad_cos": 0.075,
}

DEFAULT_THRESHOLDS = {"t_missing": 0.35, "t_present": 0.65}

_DEVIATION_CAP = 6.0
_DEVIATION_MIDPOINT = 2.5  # fused deviation mapping to score 0.5

Decision = Literal["present", "missing", "uncertain"]


def golden_stats(per_frame_signals: list[dict]) -> dict[str, dict[str, float]]:
    """Per-signal mean/std across golden frames, with std floors applied."""
    stats: dict[str, dict[str, float]] = {}
    keys = _DIRECTIONAL + _SYMMETRIC
    for key in keys:
        values = [float(s[key]) for s in per_frame_signals if key in s]
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values) if len(values) > 1 else 0.0
        std = max(math.sqrt(var), _STD_FLOORS.get(key, 0.03))
        stats[key] = {"mean": mean, "std": std}
    return stats


def fuse(signals: dict, stats: dict[str, dict[str, float]], weights: dict[str, float] | None = None) -> float:
    """Fuse per-signal deviations into a presence score in [0, 1] (1 = present).

    Each signal contributes a non-negative deviation in golden-stds; the
    weighted mean deviation is squashed through a logistic centered at
    ``_DEVIATION_MIDPOINT``.
    """
    weights = weights or DEFAULT_WEIGHTS
    total_w = 0.0
    total_dev = 0.0
    for key, w in weights.items():
        if key not in stats or key not in signals:
            continue
        mean, std = stats[key]["mean"], stats[key]["std"]
        z = (float(signals[key]) - mean) / std
        if key in _DIRECTIONAL:
            deviation = max(0.0, -z)
        else:
            deviation = abs(z)
        total_dev += w * min(deviation, _DEVIATION_CAP)
        total_w += w
    if total_w == 0:
        return 0.5
    fused_dev = total_dev / total_w
    return 1.0 / (1.0 + math.exp(fused_dev - _DEVIATION_MIDPOINT))


def decide(score: float, thresholds: dict[str, float] | None = None) -> Decision:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if score >= thresholds["t_present"]:
        return "present"
    if score <= thresholds["t_missing"]:
        return "missing"
    return "uncertain"


def vote(decisions: Iterable[Decision]) -> Decision:
    """Combine per-frame decisions for one piece.

    Auto-present requires majority present votes AND no missing vote — a
    single missing vote forces at least review (false-rejection preferred).
    """
    decisions = list(decisions)
    if not decisions:
        return "uncertain"
    if len(decisions) == 1:
        return decisions[0]
    n_present = decisions.count("present")
    n_missing = decisions.count("missing")
    majority = (len(decisions) // 2) + 1
    if n_missing == 0 and n_present >= majority:
        return "present"
    if n_missing >= majority:
        return "missing"
    return "uncertain"
