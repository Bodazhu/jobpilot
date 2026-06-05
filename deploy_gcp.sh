#!/bin/bash
# ============================================================
# JobPilot — Google Cloud Run Deployment Script
# ============================================================
# Prerequisites:
#   1. gcloud CLI installed (https://cloud.google.com/sdk/docs/install)
#   2. gcloud auth login && gcloud auth configure-docker
#   3. Indexes already built locally (run load_kaggle_data.py first)
#
# Usage:
#   chmod +x deploy_gcp.sh
#   ./deploy_gcp.sh YOUR_PROJECT_ID
# ============================================================

set -e

PROJECT_ID="${1:?Usage: ./deploy_gcp.sh PROJECT_ID}"
REGION="us-central1"
SERVICE="jobpilot"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"

echo "==> Deploying JobPilot to Cloud Run"
echo "    Project : ${PROJECT_ID}"
echo "    Region  : ${REGION}"
echo "    Image   : ${IMAGE}"
echo ""

# 1. Set project
gcloud config set project "${PROJECT_ID}"

# 2. Enable required APIs
echo "==> Enabling APIs..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com

# 3. Build image on Cloud Build (no local Docker needed)
echo "==> Building image on Cloud Build..."
gcloud builds submit --tag "${IMAGE}" .

# 4. Deploy to Cloud Run
echo "==> Deploying to Cloud Run..."
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars "SECRET_KEY=$(openssl rand -hex 32)" \
  ${ANTHROPIC_API_KEY:+--set-env-vars "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}"} \
  ${ADZUNA_APP_ID:+--set-env-vars "ADZUNA_APP_ID=${ADZUNA_APP_ID},ADZUNA_APP_KEY=${ADZUNA_APP_KEY}"}

echo ""
echo "==> Deployment complete!"
gcloud run services describe "${SERVICE}" \
  --region "${REGION}" \
  --format "value(status.url)"
