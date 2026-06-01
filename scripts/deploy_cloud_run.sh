#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-mia-prod}"
REGION="${REGION:-us-central1}"
REPOSITORY="${REPOSITORY:-mold-vision}"
SERVICE_NAME="${SERVICE_NAME:-mold-vision-api}"
RUNTIME_SA="${RUNTIME_SA:-mold-vision-runtime}"
TRAINING_SA="${TRAINING_SA:-mold-vision-training}"
MIN_INSTANCES="${MIN_INSTANCES:-1}"
MAX_INSTANCES="${MAX_INSTANCES:-3}"
CPU="${CPU:-4}"
MEMORY="${MEMORY:-16Gi}"

gcloud config set project "${PROJECT_ID}"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
UPLOAD_BUCKET="${UPLOAD_BUCKET:-${PROJECT_ID}-${PROJECT_NUMBER}-mold-uploads}"
ARTIFACT_BUCKET="${ARTIFACT_BUCKET:-${PROJECT_ID}-${PROJECT_NUMBER}-mold-artifacts}"
RUNTIME_EMAIL="${RUNTIME_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
TRAINING_EMAIL="${TRAINING_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${SERVICE_NAME}:$(date +%Y%m%d%H%M%S)"

gcloud builds submit --tag "${IMAGE}" .

gcloud run deploy "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
  --service-account="${RUNTIME_EMAIL}" \
  --execution-environment=gen2 \
  --cpu="${CPU}" \
  --memory="${MEMORY}" \
  --no-cpu-throttling \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --min-instances="${MIN_INSTANCES}" \
  --max-instances="${MAX_INSTANCES}" \
  --concurrency=1 \
  --allow-unauthenticated \
  --set-env-vars="MOLD_GCP_PROJECT=${PROJECT_ID},MOLD_GCP_REGION=${REGION},MOLD_METADATA_BACKEND=firestore,MOLD_UPLOAD_BUCKET=${UPLOAD_BUCKET},MOLD_ARTIFACT_BUCKET=${ARTIFACT_BUCKET},MOLD_VERTEX_STAGING_BUCKET=gs://${ARTIFACT_BUCKET},MOLD_ENABLE_VERTEX_TRAINING=1,MOLD_VERTEX_TRAINING_IMAGE=${IMAGE},MOLD_VERTEX_SERVICE_ACCOUNT=${TRAINING_EMAIL},MOLD_CORS_ORIGINS=https://t-efficiency.com|https://www.t-efficiency.com|https://${SERVICE_NAME}-r52omw5uhq-uc.a.run.app"

gcloud run services describe "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format='value(status.url)'
