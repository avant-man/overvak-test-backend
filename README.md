[React UI]
↓ POST /analyze
[Cloud Run] ──── lightweight, handles HTTP only
↓ forwards heavy work
[Compute Engine VM - T4 GPU]
├── yt-dlp download
├── OpenCV frame sampling
└── CLIP on GPU (torch.cuda)
↓ returns JSON result
[Cloud Run] ──── sends response back to UI

# 1. Install gcloud CLI and authenticate

gcloud auth login
gcloud config set project your-gcp-project-id

# 2. Run the setup script (takes ~3 min)

chmod +x gcp_setup.sh
./gcp_setup.sh

# 3. Copy the printed IP into your React .env

echo "VITE_API_URL=http://<EXTERNAL_IP>:8000" > frontend/.env

"We chose VideoCLIP over standard CLIP because our target actions — falling, running, pushing, hitting — are fundamentally defined by motion over time. A single frame of a child mid-fall looks identical to a child jumping. VideoCLIP's temporal encoder processes 8-frame clips through a video transformer, allowing it to distinguish these actions based on trajectory and motion flow rather than static pose."
