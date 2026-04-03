# LandCoverAI — Web App

This repository provides a full-stack web application for land-cover segmentation, bounding-box prediction, and intelligent crop recommendation using trained PyTorch models.

## Features

- 🛰️ **Satellite Image Segmentation**: 7-class land-cover classification (urban, agriculture, rangeland, forest, water, barren, unknown)
- 📦 **Bounding Box Detection**: Automatic detection and labeling of land-cover regions
- 🌾 **Intelligent Crop Recommendation**: 9-stage expert system recommending suitable crops based on land-cover analysis
- 🗺️ **Interactive Map Interface**: Click anywhere on the map to analyze satellite imagery
- 📊 **Comprehensive Analytics**: Detailed indices (MAI, SHI, ASI, CFI, MEI, AFI) and confidence scores
- 🔐 **User Authentication**: Secure login/register with session management
- 📜 **Prediction History**: Track and review past analyses

## Structure

- `backend/` — Flask API server (auth, prediction, satellite image proxy, crop recommendation)
  - `app.py` — Main Flask application with all API endpoints
  - `crop_recommender.py` — 9-stage crop recommendation engine
  - `test_recommender.py` — Test suite for crop recommendation
  - `example_crop_api.py` — API usage examples
- `frontend/` — Static web UI (login, register, map-based prediction)
- `best_model.pth` — Trained UNet++ segmentation model checkpoint
- `satellite_classifier_v4.pth` — Satellite image gate classifier
- `CROP_RECOMMENDATION.md` — Detailed crop recommendation engine documentation

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

- The app uses a MySQL database for users, sessions, and prediction history.
- The prediction endpoint uses `best_model.pth` (UNet++ with EfficientNet-B7 encoder) for segmentation.
- The crop recommendation engine analyzes land-cover data through 9 stages to recommend suitable crops.
- Satellite imagery is fetched from ArcGIS World Imagery. For heavy usage you may set `ARCGIS_API_KEY`.
- See `CROP_RECOMMENDATION.md` for detailed documentation on the crop recommendation system.

## API Endpoints

### Authentication
- `POST /api/register` — Create new user account
- `POST /api/login` — Login and get auth token
- `POST /api/logout` — Logout and invalidate token
- `GET /api/me` — Get current user info

### Prediction
- `POST /api/predict` — Upload image for segmentation
- `POST /api/predict/coordinates` — Analyze satellite image from lat/lon coordinates
- `GET /api/predictions` — Get prediction history

### Crop Recommendation
- `POST /api/recommend-crops` — Get crop recommendations from land-cover percentages

**Example Request:**
```json
{
  "percentages": {
    "urban_land": 5,
    "agriculture": 45,
    "rangeland": 15,
    "forest": 20,
    "water": 10,
    "barren": 5
  },
  "top_n": 10
}
```

**Example Response:**
```json
{
  "status": "ok",
  "water_regime": "SUB_HUMID",
  "soil_class": "PRODUCTIVE",
  "market_class": "COMMERCIAL",
  "ranked_crops": [
    {
      "rank": 1,
      "crop": "Wheat",
      "category": "Cereal",
      "season": "Rabi",
      "score": 78.5,
      "regime_match": true
    }
  ]
}
```

## Optional: Use an ArcGIS API key

Set an environment variable before running the server:

```powershell
$env:ARCGIS_API_KEY = "YOUR_API_KEY"
python backend/app.py
```
