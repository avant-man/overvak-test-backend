#!/usr/bin/env bash
set -e

PROJECT_ID="your-gcp-project-id"       # ← replace
ZONE="us-central1-a"
VM_NAME="overvak-gpu-vm"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
DISK_SIZE="50GB"
IMAGE_FAMILY="common-cu113"
IMAGE_PROJECT="deeplearning-platform-release"
FIREWALL_RULE="allow-overvak-8000"

echo "==> [1/4] Creating GPU VM"
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="type=$GPU_TYPE,count=1" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$DISK_SIZE" \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"

echo "==> [2/4] Opening port 8000"
gcloud compute firewall-rules create "$FIREWALL_RULE" \
  --project="$PROJECT_ID" \
  --allow=tcp:8000 \
  --source-ranges=0.0.0.0/0 \
  || echo "Firewall rule may already exist, continuing..."

echo "==> [3/4] Copying backend code"
gcloud compute scp --recurse ./backend "$VM_NAME":~/overvak \
  --zone="$ZONE" --project="$PROJECT_ID"

echo "==> [4/4] Installing deps and starting server"
gcloud compute ssh "$VM_NAME" \
  --zone="$ZONE" --project="$PROJECT_ID" \
  --command="
    cd ~/overvak &&
    pip install -r requirements.txt -q &&
    nohup uvicorn main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
    echo 'Server started.'
  "

EXTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" \
  --zone="$ZONE" --project="$PROJECT_ID" \
  --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo "✅ Backend live at: http://$EXTERNAL_IP:8000"
echo "👉 Add to React .env: VITE_API_URL=http://$EXTERNAL_IP:8000"
echo "🔴 Stop VM when done: gcloud compute instances stop $VM_NAME --zone=$ZONE --project=$PROJECT_ID"