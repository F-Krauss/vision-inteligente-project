"""Per-piece presence verification against registered golden sections.

This package converts "find anything missing" into per-piece binary checks at
registered locations, using lighting-invariant local signals instead of
cross-image photometry (which fails under handheld pose + lighting changes).
"""

from mold_inspection.presence.alignment import ZoneAlignment, align_zone, refine_piece
from mold_inspection.presence.embedder import PatchEmbedder, get_embedder
from mold_inspection.presence.fusion import decide, fuse, golden_stats, vote
from mold_inspection.presence.signals import edge_stats, interior_ring_stats, piece_signals

__all__ = [
    "ZoneAlignment",
    "align_zone",
    "refine_piece",
    "PatchEmbedder",
    "get_embedder",
    "golden_stats",
    "fuse",
    "decide",
    "vote",
    "interior_ring_stats",
    "edge_stats",
    "piece_signals",
]
