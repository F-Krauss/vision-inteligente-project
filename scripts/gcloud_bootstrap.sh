#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-mia-prod}"
REGION="${REGION:-us-central1}"
REPOSITORY="${REPOSITORY:-mold-vision}"
SERVICE_NAME="${SERVICE_NAME:-mold-vision-api}"
RUNTIME_SA="${RUNTIME_SA:-mold-vision-runtime}"
TRAINING_SA="${TRAINING_SA:-mold-vision-training}"

gcloud config set project "${PROJECT_ID}"

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  firestore.googleapis.com \
  aiplatform.googleapis.com \
  iamcredentials.googleapis.com

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
UPLOAD_BUCKET="${UPLOAD_BUCKET:-${PROJECT_ID}-${PROJECT_NUMBER}-mold-uploads}"
ARTIFACT_BUCKET="${ARTIFACT_BUCKET:-${PROJECT_ID}-${PROJECT_NUMBER}-mold-artifacts}"

if ! gcloud artifacts repositories describe "${REPOSITORY}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPOSITORY}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Mold vision containers"
fi

for bucket in "${UPLOAD_BUCKET}" "${ARTIFACT_BUCKET}"; do
  if ! gcloud storage buckets describe "gs://${bucket}" >/dev/null 2>&1; then
    gcloud storage buckets create "gs://${bucket}" \
      --project="${PROJECT_ID}" \
      --location="${REGION}" \
      --uniform-bucket-level-access
  fi
done

if ! gcloud firestore databases describe --database="(default)" >/dev/null 2>&1; then
  gcloud firestore databases create --database="(default)" --location=nam5 --type=firestore-native
fi

for account in "${RUNTIME_SA}" "${TRAINING_SA}"; do
  email="${account}@${PROJECT_ID}.iam.gserviceaccount.com"
  if ! gcloud iam service-accounts describe "${email}" >/dev/null 2>&1; then
    gcloud iam service-accounts create "${account}" --display-name="${account}"
  fi
done

RUNTIME_EMAIL="${RUNTIME_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
TRAINING_EMAIL="${TRAINING_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${RUNTIME_EMAIL}" \
  --role="roles/storage.objectAdmin" >/dev/null
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${RUNTIME_EMAIL}" \
  --role="roles/datastore.user" >/dev/null
gcloud iam service-accounts add-iam-policy-binding "${RUNTIME_EMAIL}" \
  --member="serviceAccount:${RUNTIME_EMAIL}" \
  --role="roles/iam.serviceAccountTokenCreator" >/dev/null

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${TRAINING_EMAIL}" \
  --role="roles/storage.objectAdmin" >/dev/null
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${TRAINING_EMAIL}" \
  --role="roles/aiplatform.user" >/dev/null

cat <<EOF
Bootstrap complete.
PROJECT_ID=${PROJECT_ID}
REGION=${REGION}
SERVICE_NAME=${SERVICE_NAME}
UPLOAD_BUCKET=${UPLOAD_BUCKET}
ARTIFACT_BUCKET=${ARTIFACT_BUCKET}
RUNTIME_SERVICE_ACCOUNT=${RUNTIME_EMAIL}
TRAINING_SERVICE_ACCOUNT=${TRAINING_EMAIL}
EOF
