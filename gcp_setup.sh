#!/usr/bin/env bash
set -e

PROJECT_ID="project-64200d5f-0e35-4776-965"
VM_NAME="overvak-gpu-vm"
FIREWALL_RULE="allow-overvak-8000"
OPS_AGENT_LABEL="v2-template-1-5-0"
SNAPSHOT_POLICY="default-schedule-1"

# ──────────────────────────────────────────────
# Usage:
#   ./gcp_setup.sh <zone>        e.g.  ./gcp_setup.sh us-central1-c
#   ./gcp_setup.sh               (will prompt)
#
# Create the VM on the GCP dashboard first, then run this script to
# configure the Ops Agent, snapshot schedule, firewall, and deploy code.
# ──────────────────────────────────────────────

if [[ -n "$1" ]]; then
  ZONE="$1"
else
  read -r -p "Enter the zone your VM was created in (e.g. us-central1-c): " ZONE
fi

if [[ -z "$ZONE" ]]; then
  echo "ERROR: zone is required."
  exit 1
fi

REGION="${ZONE%-*}"
OPS_POLICY_NAME="goog-ops-agent-${OPS_AGENT_LABEL}-${ZONE}"

echo "    VM name : $VM_NAME"
echo "    Zone    : $ZONE"
echo "    Region  : $REGION"

echo "==> [1/5] Installing Ops Agent policy"
printf 'agentsRule:\n  packageState: installed\n  version: latest\ninstanceFilter:\n  inclusionLabels:\n  - labels:\n      goog-ops-agent-policy: %s\n' \
  "$OPS_AGENT_LABEL" > /tmp/ops-config.yaml
gcloud compute instances ops-agents policies create "$OPS_POLICY_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --file=/tmp/ops-config.yaml \
  || echo "    Ops Agent policy may already exist, continuing..."

echo "==> [2/5] Creating snapshot schedule"
gcloud compute resource-policies create snapshot-schedule "$SNAPSHOT_POLICY" \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --max-retention-days=14 \
  --on-source-disk-delete=keep-auto-snapshots \
  --daily-schedule \
  --start-time=01:00 \
  || echo "    Snapshot policy may already exist, continuing..."

gcloud compute disks add-resource-policies "$VM_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --resource-policies="projects/$PROJECT_ID/regions/$REGION/resourcePolicies/$SNAPSHOT_POLICY"

echo "==> [3/5] Opening port 8000"
gcloud compute firewall-rules create "$FIREWALL_RULE" \
  --project="$PROJECT_ID" \
  --allow=tcp:8000 \
  --source-ranges=0.0.0.0/0 \
  || echo "    Firewall rule may already exist, continuing..."

echo "==> [4/5] Copying backend code"
gcloud compute scp --recurse ./backend "$VM_NAME":~/overvak \
  --zone="$ZONE" --project="$PROJECT_ID"

echo "==> [5/5] Installing deps and starting server"
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
