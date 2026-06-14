# vision-inteligente-project (mold-inspection)

Two parts:
- Python package `mold-inspection` (pyproject, requires-python >=3.10) — guided visual inspection
  pipeline for industrial molds. Cloud extras: FastAPI + uvicorn, google-cloud-storage/firestore/
  aiplatform, google-genai.
- Vite + React frontend (lab UI, presence detection, validation sessions, benchmarks) + Supabase.

## Infra
- Dockerfile + `cloudbuild.yaml` → Cloud Build / Cloud Run. GCloud-deployed.

## Commands
- Python tests: `npm run test` wraps them (check package.json). Install cloud deps: `pip install -e ".[cloud]"`.
- Frontend: Vite dev/build (see package.json scripts).

## Notes
- Keep the Python pipeline and the React lab UI in sync on the data contract.
