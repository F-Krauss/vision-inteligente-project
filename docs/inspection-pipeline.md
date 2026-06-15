# Inspection Pipeline & Detection Methods

How `CloudInspectionPipeline.inspect()` (`mold_inspection/cloud/pipeline.py`) decides
`correct` / `review` / `retake_photo` for a captured zone photo. Inspection is
**deterministic** (classical CV + YOLO; no LLM) and biased toward **false-rejection over
false-approval** — anything ambiguous goes to `review`, never a confident wrong approval.

Images are cellphone photos (~4032px wide) of zones holding ~106 parts (pistons / bolts /
pins), the smallest ~15px. Every design choice below exists to catch those small parts
without auto-approving a faulty mold.

## Flow overview

```
capture-quality gate ──fail──▶ retake_photo
        │ ok
mold segmentation/crop ──fail──▶ retake_photo
        │ ok
   has promoted model?
   ├─ no  ─▶ reference path  (no trained model yet)
   └─ yes ─▶ trained path    (anomaly ∧ YOLO ∧ reference cross-check)
```

1. **Capture quality** (`evaluate_capture_quality`) — blur / brightness / resolution.
2. **Mold segmentation** (`normalize_mold_crop`) — locate + crop the mold; reject framing
   the comparison can't use.
3. **Branch** on whether a promoted model exists at
   `model_registry/<family>/<zone>/best_model/profile.json`.

## No-model path — reference inspection

`_inspect_reference_pieces` tries three methods in order; the first that can compare wins:

### 1. Multi-annotated-reference per-part consensus (primary)

`piece_inspector.inspect_expected_pieces_against_references`, fed by
`references.gather_annotated_references` (every annotated golden image for the zone, not
just the latest).

- A **canonical part set** is the union of all references' annotation boxes (by `element_id`).
- Each annotated golden casts **one vote per part** via the existing ECC-aligned per-ROI
  diff (`inspect_expected_pieces_against_reference`).
- Votes combine with a **false-rejection-safe three-state rule**:

  | votes flagging the part `missing` | verdict |
  | --- | --- |
  | strict majority of references (2-of-2, 2-of-3) | **missing** |
  | some but not a majority (or any inconclusive) | **uncertain → review** |
  | none | **present** |

  The zone is auto-approved only when **every** part is `present`. A genuinely missing part
  differs from *all* golden references, so it earns full support; a single-reference
  lighting/pose phantom never silently approves *or* confidently fails — it goes to review.
  With one reference this degrades to plain single-reference behavior.

This is the most precise path for tiny parts because it inspects each known part **ROI**
rather than a global image blob. Findings report `support` / `support_ratio`.

### 2. Classical CV consensus (coarse fallback)

`cv_inspector.inspect_with_cv_consensus` — ORB+SSIM diff of the candidate against every
golden reference inside one hardcoded ROI polygon, keeping only findings corroborated by a
majority of references. Used when the zone has golden references but no per-part annotations.

### 3. Legacy single-reference pixel diff (last resort)

`inspect_expected_pieces_against_reference` against the latest reference — only reached when
the consensus paths can't compare at all.

## Trained path — model + agreement gate

When a promoted model exists, three **independent** signals must agree to auto-approve:

1. **PatchCore anomaly** (`model_suite.inspect_best_model`) — whole-zone anomaly score/heatmap.
2. **YOLO per-part detector** (`piece_inspector.inspect_expected_pieces`) — expected pieces
   matched to detections by class × ROI; an expected piece with no detection at its ROI is
   `missing`.
3. **Reference cross-check** (`pipeline._reference_cross_check`) — the annotated-reference
   consensus (or CV consensus when there are no annotations).

The result is `correct` only if **all three agree**; any disagreement downgrades to `review`
and attaches `result["reference_cross_check"]`. Requiring consensus across independent
signals drives false-approval toward zero.

## Small-part handling

- **Inference resolution** — `inspect_expected_pieces` predicts at `imgsz=1280` (vs the
  ultralytics default 640) with **tiled inference on by default** (`_tiled_predict`:
  overlapping full-resolution tiles → IoU-NMS merge). At 640 a 15px part downscales to ~2px
  and disappears; tiling predicts each region at native scale. Disable via `tile=False` for a
  latency-constrained target.
- **Training defaults** — `yolo_runtime.train_yolo` defaults to `yolo11s` @ `imgsz=1280`
  (was `yolo11n` @ 960) to match inference. Override per zone via the train CLI.
- **ROI-relative change floor** — `_localized_changes` sizes its minimum change against the
  (tight, per-part) ROI, not the 12 MP frame, so a ~15px change isn't rejected before it can
  be flagged.

## Measuring accuracy — `scripts/eval_inspection.py`

"Accurate enough" is a number. The benchmark files in `benchmarks/annotations/*.json` are a
labeled held-out set: `golden_images` (references), `pieces` (canonical boxes), and
`eval_images` with `missing_piece_ids` ground truth (empty = a true OK).

```bash
python3 -m scripts.eval_inspection                      # all benchmarks
python3 -m scripts.eval_inspection --glob 'try-photos-mold-a*'
python3 -m scripts.eval_inspection --report reports/eval.md --max-false-approval 0.0
```

Reports the metrics that matter and **hard-gates** on false-approval (exit non-zero if
exceeded, so it can gate CI):

- `false_approval_rate` — faulty mold predicted `correct` (**target 0**, the hard gate).
- `false_rejection_rate` — good mold sent to review (operator load).
- `auto_approval_rate` — good molds auto-approved without review.
- `piece_recall` — fraction of known-missing parts actually flagged.

Target: **false_approval_rate = 0 at an acceptable auto_approval_rate**. Tune
consensus strength (`min_support`) and the per-ROI diff thresholds against this set.

## History

The previous Gemini Vision fallback was removed 2026-06-14 (deterministic CV + multi-
reference consensus is more accurate and reproducible for this pass/fail task; ambiguity
defers to humans rather than an LLM). See `cv_inspector.py` (formerly `gemini_inspector.py`).
