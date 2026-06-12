# Viability assessment — per-part mold verification with tablet photos

Date: 2026-06-12. Based on the try-photos-mold-a and try-photos-last-mold datasets
(12 sections, ~190 photos), the IMG_2440 manual registration (106 parts), and the
presence package (alignment + signals + fusion).

## Verdict

**Viable, with one hard requirement: pose-controlled capture.** The detection
physics works — a removed piece is clearly distinguishable at its registered
location when the inspection photo is taken from (approximately) the registered
pose. Free-pose photos make small pieces (pins, screws — under ~2% of frame)
undetectable by any method tested, because single-homography alignment of a
deeply 3D mold leaves 30–50 px of parallax error, and per-piece local refinement
slides onto identical-looking neighboring structures.

## Evidence

What works (verified on data):

1. Manual registration is extractable and transferable. The user's annotated
   IMG_2440 (green zone + 106 red part outlines) was extracted programmatically
   and projected onto other photos of the same mold at 0.78–0.88 inlier ratio,
   with every box landing on its part (`scripts/extract_manual_annotation.py`,
   `benchmarks/registrations/`).
2. Same-pose pair comparison localizes real removals, even tiny ones. The
   consecutive pair IMG_2572/2573 cleanly exposed a removed pin (bright pin head
   → empty dark hole) at (0.36, 0.62) in a ~30 px box.
3. Zone discipline removes the false-positive class. With hand-drawn mold-body
   zones (`benchmarks/zones.json`), floor/people/background blobs are filtered;
   present-only photos scored zero false missing for medium parts.
4. Medium/large parts (brackets, bronze pads, blocks) verify correctly across
   moderate pose change.

What fails (verified on data):

1. Cross-pose verification of small parts. The same registered pin box, mapped
   into five different present photos via golden→photo homography, produced
   interior/ring ratios from 0.006 to 1.59 — i.e. the box lands on the wrong
   structure more often than not. All per-piece scores degenerate accordingly.
2. Pair-diff ground truth without zone+verification is contaminated: most diff
   blobs are parallax/shadow artifacts (background molds, edge slivers), which
   out-score the true 30 px removal signal.
3. Whole-scene approaches (Gemini single/side-by-side/few-shot, global SSIM/tile
   diff) were already shown unreliable in earlier sessions for this same reason.

## Required protocol (what makes it viable)

1. **Guided capture on the tablet**: at registration, store each section's golden
   photo; at inspection, show it as a translucent ghost overlay and require the
   operator to match it before shooting. Within a few degrees, every comparison
   becomes the consecutive-pair case — which demonstrably works down to pin-head
   scale. A guided video sweep is an alternative: pan slowly, auto-select the
   frames nearest each registered pose.
2. **One-time registration per mold section** by the operator: draw the zone and
   outline each part (the IMG_2440 methodology). 10–15 minutes per mold, done once.
3. **2–3 shots per section per inspection** for multi-frame voting
   (`fusion.vote`: one missing vote forces review — false rejection preferred).
4. **Section granularity for small parts**: capture distance such that the
   smallest registered part is at least ~2% of the frame (the dataset's "detail"
   sections already do this).

## What cannot be fixed in software alone

Free-pose, full-mold photos for pin-sized parts. The information is simply not
reliably recoverable: repetitive machined patterns defeat local matching, and
3D parallax defeats global alignment. Pose control at capture is cheaper and
strictly more effective than any further algorithm work.

## Validation still pending (needs new photos from the factory)

A re-shot set following the protocol: per section, golden + present + missing
shots taken with the ghost-overlay discipline. The presence pipeline should then
be benchmarked with `scripts/presence_check_registry.py` and gated on:
zero missed removals (no false "present"), false-review rate under ~10%.
