# Mold Vision App Workflow

This document describes the current working process of the app and the target flow for annotation-driven inspection.

> For how a captured photo is actually scored (`correct` / `review` / `retake_photo`) — the
> reference consensus, the trained-model agreement gate, small-part handling, and the
> accuracy eval — see [inspection-pipeline.md](inspection-pipeline.md).

## Core Goal

The app should let a user create a mold, collect training photos by section, annotate the visible pieces, train a model from those annotations, choose a golden sample for each section, and then validate a mold section by section using guided capture and missing-piece detection.

## Current App Process

1. Create or select a mold in `Moldes`.
2. Upload training images for the mold and zone.
3. Open `Anotar`.
4. Upload or select an image for the mold zone.
5. Draw boxes around expected pieces.
6. Use `Auto-anotar borrador` to create initial boxes from configured ROIs when the zone has expected-piece regions.
7. Correct labels, box positions, and status (`present`, `missing`, `uncertain`).
8. Save annotations.
9. Generate a YOLO-style dataset from annotations.
10. Start inspector training from the generated dataset. In local prototype mode, this writes a promoted JSON model artifact from the latest corrected annotation boxes.
11. `Auto-anotar borrador` can then draft boxes from that promoted local model artifact before falling back to templates or ROIs.
12. Upload/select a golden sample in `Captura` for the current `family + zone_id`.
13. Capture a new section photo with the golden sample overlay.
14. The API validates image quality and alignment.
15. The inspector evaluates the section and returns `correct`, `review`, or `retake_photo`.
16. `Validaciones` shows leadership metrics, retakes, model health, and validations by mold/zone.

## Target Production Flow

1. User creates a new mold.
2. User defines how the mold is separated into zones and views.
3. Each zone can have `left`, `right`, and/or `front` views.
4. If the user provides measures and enough photos, the program can suggest a zone/view split, but the user confirms it.
5. User uploads images for each required zone view.
6. The system normalizes images into a consistent orientation, size, and crop for training.
7. System creates training versions per `mold + zone + view`.
8. System proposes annotations automatically when possible.
9. User corrects or completes annotations.
10. User marks whether each visible expected piece is present, missing, or uncertain.
11. System creates the dataset from annotations for each zone view.
12. Model training starts per zone view, not only per whole mold.
13. If the model needs stronger false-pass protection, user uploads missing-piece or fault images for the affected zone view.
14. User selects the golden sample for each zone view.
15. Operator selects a mold for validation.
16. App guides the operator through every required zone view.
17. For each zone view, app shows golden sample overlay and capture guidance.
18. Operator captures the photo for that zone view.
19. App validates image quality, alignment, and section identity.
20. Model checks annotations/expected pieces and flags missing pieces.
21. App records each section result in a mold validation session.
22. Server blocks final mold completion until every required zone view has a passed or reviewed photo.
23. Final mold validation is complete only when all registered zone views are `correct` or reviewed by a supervisor.

## Zone And View Split

A mold is not one single image. It is a set of required zone views.

Example:

```text
Mold A
  Zone 1
    left
    right
  Zone 2
    left
    right
  Zone 3
    front
```

The app now has a section planner in `Captura` where the user can choose the number of zones and whether each zone needs left, right, and/or front views. The selected split creates required section IDs such as:

```text
zona_01_left
zona_01_right
zona_02_left
zona_02_right
zona_03_front
```

Each generated zone view needs its own:

- training images
- annotations
- normalized dataset output
- golden sample
- model/training status
- capture result during validation

When the mold is used in production, the system should require one valid photo for every required zone view before the complete mold is considered passed.

The section plan is persisted through `/v1/mold-section-plans/{mold_key}` and cached locally in the browser. Saving the plan also registers each generated zone view in `/v1/zones`, so capture, references, annotations, datasets, and training can address the same `family + zone_id` contract.

Runtime mold validation is tracked through `/v1/mold-validation-sessions`. A session is tied to the persisted section plan. Each section result is posted to `/v1/mold-validation-sessions/{session_id}/sections/{section_id}`. The server marks the mold `complete` only when every required section has a `correct` or `review` result; `retake_photo` keeps that section missing.

## Annotation Model Rule

Annotations are the center of the system.

- The expected piece list comes from `config/inspection.json`.
- `present` boxes teach the model what normal visible pieces look like.
- `missing` and `uncertain` labels are useful for review state and future model calibration.
- Missing-piece images are not always required for a first detector, because expected pieces can be inferred as missing when they are not detected in their expected section.
- Missing-piece/fault images are still recommended when the business risk is false-pass. They improve thresholds and help the model learn real missing-piece cases.

## Future Auto-Annotation

The auto-annotation loop should work like this:

1. User uploads images for a new mold zone view.
2. System normalizes the image to the training orientation for that zone view.
3. Current promoted model for that zone view predicts piece boxes.
4. App saves predictions as draft annotations, not final truth.
5. User corrects labels, boxes, and missing/uncertain status.
6. Corrected annotations are added to the dataset.
7. Training uses corrected data to improve the next model.
8. New model auto-annotates future molds with better accuracy.

The app now calls `/v1/annotations/auto-draft` from `Anotar`. The endpoint returns draft boxes in this order:

- `model`: use the promoted model for `family + zone_id` when a readable local model artifact exists.
- `annotation_template`: reuse the latest corrected annotation for the same zone as a starting template.
- `roi`: use configured expected-piece ROIs.
- `empty`: return no boxes when none of the above exists.

All auto-annotations are drafts. User correction is still required before saving ground truth.

Current local training writes an `annotation_template_detector` JSON artifact for the promoted inspector candidate. That artifact is not a production neural network; it is a working local bridge so the UI can prove the annotation -> dataset -> training -> promoted model -> auto-annotation loop.

## Section Completion

Each mold needs a section checklist.

For each zone view:

- Golden sample exists.
- Expected pieces are configured.
- Training images exist.
- Annotations are saved.
- Dataset is generated.
- Model is trained or marked as pending review.
- Capture has been completed.
- Inspection result is accepted or reviewed.

The mold is fully validated only after all registered sections pass this checklist.

## Current Gaps

- Roles are not enforced yet.
- Auto-annotation can use the local promoted JSON artifact, but production neural-network inference still needs real model artifacts from training jobs.
- Training job status is created, but real training orchestration still needs production integration.
- UI now syncs section completion to the server, but a supervisor review role and final signed approval screen are still needed.
