"""Synthetic tests for the per-piece presence engine.

A textured "plate" with bright textured pieces is rendered; the candidate
erases one piece (open dark socket), is re-shot with a known homography and a
global brightness change. The engine must recover geometry, keep present
pieces present, and flag the erased piece — without any real model download
(a deterministic fake embedder keeps tests hermetic).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mold_inspection.presence.alignment import align_zone, refine_piece, warp_to_golden
from mold_inspection.presence.fusion import (
    DEFAULT_THRESHOLDS,
    decide,
    fuse,
    golden_stats,
    vote,
)
from mold_inspection.presence.signals import interior_ring_stats, piece_signals

RNG = np.random.default_rng(7)

H_IMG, W_IMG = 900, 1200

PIECES = [
    (260, 220, 360, 300),
    (520, 200, 650, 320),
    (300, 520, 420, 640),
    (700, 500, 840, 580),
]

ZONE = [(0.12, 0.12), (0.88, 0.12), (0.88, 0.88), (0.12, 0.88)]


class FakeEmbedder:
    """Deterministic stand-in: downsampled CLAHE grayscale, L2-normalized."""

    def embed(self, patches_bgr):
        out = []
        for patch in patches_bgr:
            if patch.size == 0:
                patch = np.zeros((16, 16, 3), dtype=np.uint8)
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
            vec = cv2.resize(gray, (16, 16)).astype(np.float32).ravel()
            vec -= vec.mean()
            norm = np.linalg.norm(vec)
            out.append(vec / norm if norm > 0 else vec)
        return np.stack(out)


def _draw_scene(missing_idx: int | None = None) -> np.ndarray:
    # Deterministic per call: golden and candidate must share every texture
    # except the erased piece, like consecutive photos of one physical mold.
    rng = np.random.default_rng(7)
    base = rng.integers(70, 110, (H_IMG, W_IMG), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 1.2)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    # Texture anchors so SIFT has plenty to match.
    pts = rng.integers(0, [W_IMG, H_IMG], (320, 2))
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 6)), tuple(int(v) for v in rng.integers(30, 200, 3)), -1)

    for idx, (x1, y1, x2, y2) in enumerate(PIECES):
        # Socket rim (always drawn).
        cv2.rectangle(img, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), (50, 50, 52), 6)
        if idx == missing_idx:
            # Open socket: dark cavity with faint noise.
            cavity_rng = np.random.default_rng(200 + idx)
            cavity = cavity_rng.integers(8, 26, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
            img[y1:y2, x1:x2] = cavity
        else:
            # Installed piece: bright metallic block with texture.
            block_rng = np.random.default_rng(100 + idx)
            block = block_rng.integers(150, 205, (y2 - y1, x2 - x1), dtype=np.uint8)
            block = cv2.GaussianBlur(block, (0, 0), 0.8)
            img[y1:y2, x1:x2] = cv2.cvtColor(block, cv2.COLOR_GRAY2BGR)
            cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 9, (90, 90, 95), -1)
    return img


def _reshoot(img: np.ndarray, brightness: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Apply a known homography (slight rotation/translation/perspective) + exposure change."""
    angle = 2.5
    h_rot = cv2.getRotationMatrix2D((W_IMG / 2, H_IMG / 2), angle, 1.02)
    h3 = np.vstack([h_rot, [0, 0, 1]]).astype(np.float64)
    h3[0, 2] += 18
    h3[1, 2] -= 12
    h3[2, 0] = 4e-6
    warped = cv2.warpPerspective(img, h3, (W_IMG, H_IMG), borderMode=cv2.BORDER_REPLICATE)
    warped = np.clip(warped.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return warped, h3


def _signals_for(golden, candidate_warped_back, embedder):
    return [
        piece_signals(golden, candidate_warped_back, {"bbox_px": bbox}, embedder)
        for bbox in PIECES
    ]


@pytest.fixture(scope="module")
def golden():
    return _draw_scene(missing_idx=None)


@pytest.fixture(scope="module")
def embedder():
    return FakeEmbedder()


@pytest.fixture(scope="module")
def stats(golden, embedder):
    frames = []
    for jitter_b in (1.0, 0.94, 1.06):
        noisy = np.clip(golden.astype(np.float32) * jitter_b + RNG.normal(0, 2, golden.shape), 0, 255).astype(np.uint8)
        frames.append(noisy)
    per_piece = [[] for _ in PIECES]
    for frame in frames:
        for i, sig in enumerate(_signals_for(golden, frame, embedder)):
            per_piece[i].append(sig)
    return [golden_stats(sigs) for sigs in per_piece]


def test_alignment_recovers_known_homography(golden):
    candidate, _ = _reshoot(golden, brightness=0.8)
    alignment = align_zone(golden, candidate, ZONE)
    assert alignment.ok, alignment.reason
    assert alignment.inliers >= 30
    assert alignment.reproj_px < 3.0

    back = warp_to_golden(candidate, alignment, golden.shape)
    core = (slice(100, H_IMG - 100), slice(100, W_IMG - 100))
    g = cv2.cvtColor(golden[core], cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(back[core], cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Brightness differs (0.8x); correlation must still be near-perfect.
    corr = np.corrcoef(g.ravel(), b.ravel())[0, 1]
    assert corr > 0.9


def test_missing_piece_detected_under_pose_and_lighting_change(golden, embedder, stats):
    missing_idx = 1
    candidate = _draw_scene(missing_idx=missing_idx)
    candidate, _ = _reshoot(candidate, brightness=0.72)

    alignment = align_zone(golden, candidate, ZONE)
    assert alignment.ok, alignment.reason
    back = warp_to_golden(candidate, alignment, golden.shape)

    sigs = _signals_for(golden, back, embedder)
    scores = [fuse(s, st) for s, st in zip(sigs, stats)]
    decisions = [decide(s, DEFAULT_THRESHOLDS) for s in scores]

    assert decisions[missing_idx] == "missing", f"scores={scores}"
    for i, d in enumerate(decisions):
        if i != missing_idx:
            assert d == "present", f"piece {i} score={scores[i]}"
    assert scores[missing_idx] < min(s for i, s in enumerate(scores) if i != missing_idx)


def test_all_present_stays_present_when_reshot(golden, embedder, stats):
    candidate, _ = _reshoot(golden, brightness=1.3)
    alignment = align_zone(golden, candidate, ZONE)
    assert alignment.ok, alignment.reason
    back = warp_to_golden(candidate, alignment, golden.shape)

    sigs = _signals_for(golden, back, embedder)
    scores = [fuse(s, st) for s, st in zip(sigs, stats)]
    for i, score in enumerate(scores):
        assert decide(score) == "present", f"piece {i} score={score} sigs={sigs[i]}"


def test_ring_ratio_is_exposure_invariant(golden):
    gray = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    dimmed = np.clip(gray.astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
    for bbox in PIECES:
        r1 = interior_ring_stats(gray, bbox)["ring_ratio"]
        r2 = interior_ring_stats(dimmed, bbox)["ring_ratio"]
        assert abs(r1 - r2) / max(r1, 1e-6) < 0.06


def test_refine_piece_locks_socket_even_when_piece_missing(golden):
    candidate = _draw_scene(missing_idx=2)
    g_gray = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    c_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    shifted = np.roll(c_gray, (6, -9), axis=(0, 1))
    dx, dy, peak = refine_piece(g_gray, shifted, PIECES[2])
    # Context (rim + plate) should still register the shift despite the void.
    assert abs(dx - (-9)) <= 3 and abs(dy - 6) <= 3, (dx, dy, peak)


def test_vote_policy_prefers_false_rejection():
    assert vote(["present", "present", "present"]) == "present"
    assert vote(["present", "present", "missing"]) == "uncertain"
    assert vote(["missing", "missing", "present"]) == "missing"
    assert vote(["present", "present", "uncertain"]) == "present"
    assert vote(["uncertain", "uncertain", "present"]) == "uncertain"
    assert vote(["missing"]) == "missing"
    assert vote([]) == "uncertain"
