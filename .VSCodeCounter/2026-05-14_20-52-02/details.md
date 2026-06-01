# Details

Date : 2026-05-14 20:52:02

Directory /Users/fukunugget/dev/vision-inteligente-project

Total : 54 files,  57705 codes, 5 comments, 1498 blanks, all 59208 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [Dockerfile](/Dockerfile) | Docker | 19 | 0 | 2 | 21 |
| [README.md](/README.md) | Markdown | 247 | 0 | 79 | 326 |
| [cloudbuild.yaml](/cloudbuild.yaml) | YAML | 13 | 0 | 2 | 15 |
| [config/inspection.json](/config/inspection.json) | JSON | 39 | 0 | 1 | 40 |
| [data/README\_DATASET.md](/data/README_DATASET.md) | Markdown | 51 | 0 | 18 | 69 |
| [mold\_inspection/\_\_init\_\_.py](/mold_inspection/__init__.py) | Python | 2 | 1 | 2 | 5 |
| [mold\_inspection/anomaly\_reference.py](/mold_inspection/anomaly_reference.py) | Python | 531 | 0 | 83 | 614 |
| [mold\_inspection/cli.py](/mold_inspection/cli.py) | Python | 384 | 0 | 35 | 419 |
| [mold\_inspection/cloud/\_\_init\_\_.py](/mold_inspection/cloud/__init__.py) | Python | 2 | 1 | 3 | 6 |
| [mold\_inspection/cloud/app.py](/mold_inspection/cloud/app.py) | Python | 217 | 0 | 41 | 258 |
| [mold\_inspection/cloud/config.py](/mold_inspection/cloud/config.py) | Python | 48 | 0 | 6 | 54 |
| [mold\_inspection/cloud/datasets.py](/mold_inspection/cloud/datasets.py) | Python | 176 | 0 | 31 | 207 |
| [mold\_inspection/cloud/guidance.py](/mold_inspection/cloud/guidance.py) | Python | 140 | 0 | 24 | 164 |
| [mold\_inspection/cloud/inspector\_training.py](/mold_inspection/cloud/inspector_training.py) | Python | 151 | 0 | 18 | 169 |
| [mold\_inspection/cloud/pipeline.py](/mold_inspection/cloud/pipeline.py) | Python | 226 | 0 | 24 | 250 |
| [mold\_inspection/cloud/public\_datasets.py](/mold_inspection/cloud/public_datasets.py) | Python | 129 | 0 | 22 | 151 |
| [mold\_inspection/cloud/recipes.py](/mold_inspection/cloud/recipes.py) | Python | 77 | 0 | 15 | 92 |
| [mold\_inspection/cloud/schemas.py](/mold_inspection/cloud/schemas.py) | Python | 236 | 0 | 61 | 297 |
| [mold\_inspection/cloud/segmenter.py](/mold_inspection/cloud/segmenter.py) | Python | 112 | 0 | 15 | 127 |
| [mold\_inspection/cloud/segmenter\_trainer.py](/mold_inspection/cloud/segmenter_trainer.py) | Python | 47 | 0 | 12 | 59 |
| [mold\_inspection/cloud/storage.py](/mold_inspection/cloud/storage.py) | Python | 189 | 0 | 25 | 214 |
| [mold\_inspection/cloud/store.py](/mold_inspection/cloud/store.py) | Python | 107 | 0 | 28 | 135 |
| [mold\_inspection/cloud/trainer.py](/mold_inspection/cloud/trainer.py) | Python | 94 | 0 | 20 | 114 |
| [mold\_inspection/cloud/training.py](/mold_inspection/cloud/training.py) | Python | 105 | 0 | 8 | 113 |
| [mold\_inspection/dataset.py](/mold_inspection/dataset.py) | Python | 193 | 0 | 38 | 231 |
| [mold\_inspection/decision.py](/mold_inspection/decision.py) | Python | 85 | 0 | 24 | 109 |
| [mold\_inspection/golden\_reference.py](/mold_inspection/golden_reference.py) | Python | 385 | 0 | 71 | 456 |
| [mold\_inspection/model\_suite.py](/mold_inspection/model_suite.py) | Python | 382 | 0 | 45 | 427 |
| [mold\_inspection/models.py](/mold_inspection/models.py) | Python | 158 | 0 | 37 | 195 |
| [mold\_inspection/mold\_segmenter.py](/mold_inspection/mold_segmenter.py) | Python | 271 | 0 | 27 | 298 |
| [mold\_inspection/piece\_inspector.py](/mold_inspection/piece_inspector.py) | Python | 90 | 0 | 11 | 101 |
| [mold\_inspection/yolo\_export.py](/mold_inspection/yolo_export.py) | Python | 65 | 0 | 13 | 78 |
| [mold\_inspection/yolo\_runtime.py](/mold_inspection/yolo_runtime.py) | Python | 65 | 0 | 15 | 80 |
| [scripts/deploy\_cloud\_run.sh](/scripts/deploy_cloud_run.sh) | Shell Script | 40 | 1 | 6 | 47 |
| [scripts/gcloud\_bootstrap.sh](/scripts/gcloud_bootstrap.sh) | Shell Script | 69 | 1 | 13 | 83 |
| [tests/test\_anomaly\_reference.py](/tests/test_anomaly_reference.py) | Python | 115 | 0 | 23 | 138 |
| [tests/test\_cli.py](/tests/test_cli.py) | Python | 84 | 0 | 12 | 96 |
| [tests/test\_cloud\_api.py](/tests/test_cloud_api.py) | Python | 350 | 0 | 54 | 404 |
| [tests/test\_config.py](/tests/test_config.py) | Python | 26 | 0 | 6 | 32 |
| [tests/test\_decision.py](/tests/test_decision.py) | Python | 48 | 0 | 13 | 61 |
| [tests/test\_golden\_reference.py](/tests/test_golden_reference.py) | Python | 126 | 0 | 29 | 155 |
| [tests/test\_model\_suite.py](/tests/test_model_suite.py) | Python | 120 | 0 | 15 | 135 |
| [tests/test\_split.py](/tests/test_split.py) | Python | 25 | 0 | 6 | 31 |
| [web/.env](/web/.env) | Dotenv | 3 | 0 | 1 | 4 |
| [web/index.html](/web/index.html) | HTML | 12 | 0 | 1 | 13 |
| [web/package-lock.json](/web/package-lock.json) | JSON | 1,995 | 0 | 1 | 1,996 |
| [web/package.json](/web/package.json) | JSON | 24 | 0 | 1 | 25 |
| [web/src/main.tsx](/web/src/main.tsx) | TypeScript JSX | 1,230 | 0 | 108 | 1,338 |
| [web/src/styles.css](/web/src/styles.css) | CSS | 777 | 0 | 140 | 917 |
| [web/src/utils/supabase.ts](/web/src/utils/supabase.ts) | TypeScript | 4 | 0 | 3 | 7 |
| [web/src/vite-env.d.ts](/web/src/vite-env.d.ts) | TypeScript | 0 | 1 | 1 | 2 |
| [web/tsconfig.json](/web/tsconfig.json) | JSON with Comments | 21 | 0 | 1 | 22 |
| [web/vite.config.ts](/web/vite.config.ts) | TypeScript | 13 | 0 | 2 | 15 |
| [yolov8n-seg.pt](/yolov8n-seg.pt) | XML | 47,587 | 0 | 206 | 47,793 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)