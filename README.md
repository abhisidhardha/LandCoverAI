# LandCoverAI — Web App

This repository provides a full-stack web application for land-cover segmentation and bounding-box prediction using a trained PyTorch model.

## Structure

- `backend/` — Flask API server (auth, prediction, satellite image proxy)
- `frontend/` — Static web UI (login, register, map-based prediction)
- `best_model.pth` — Trained PyTorch model checkpoint used for inference

## Getting Started

### 1) Create a Python virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

### 2) Install dependencies

```powershell
pip install -r backend/requirements.txt
```

### 3) Run the server

```powershell
python backend/app.py
```

Then open: **http://localhost:5000**

## Notes

- The app uses an SQLite database at `backend/app.db` for users and sessions.
- The prediction endpoint uses `best_model.pth` in the repository root.
- Satellite imagery is fetched from ArcGIS World Imagery. For heavy usage you may set `ARCGIS_API_KEY`.

## Optional: Use an ArcGIS API key

Set an environment variable before running the server:

```powershell
$env:ARCGIS_API_KEY = "YOUR_API_KEY"
python backend/app.py
```
