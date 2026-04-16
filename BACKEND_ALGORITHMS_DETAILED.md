# LandCoverAI: Detailed Backend Algorithms

This document provides a low-level, algorithmic breakdown of the four core "brains" of the LandCoverAI backend. These represent the transition from raw satellite data to expert agricultural advice.

---

## Algorithm 1: Dual-Stage Vision & Segmentation
**File**: `backend/app.py` (via `BBoxEngine`)

This algorithm converts raw 512x512 pixels into a structured semantic mask and bounding boxes.

### Step-by-Step Logic:
1.  **Stage 1: The Satellite Gate (Safety Filter)**
    *   **Input**: RGB tensor `I` (224x224).
    *   **Model**: EfficientNet-B0 (Binary Classifier).
    *   **Inference**: `score ← σ(model(I))`.
    *   **Decision**: IF `score < 0.5` → **HALT** (Reason: "Invalid Image").
2.  **Stage 2: UNet++ Semantic Segmentation**
    *   **Input**: Standardized RGB tensor (512x512).
    *   **Model**: UNet++ with EfficientNet-B7 encoder.
    *   **Inference**: `logits ← model(I)`.
    *   **Softmax**: `probs ← F.softmax(logits, dim=1)`.
    *   **Argmax**: `mask ← probs.argmax(axis=0)`.
3.  **Stage 3: Bounding Box & Cleanup**
    *   **Connectivity**: For each class `C` in {0...5}:
        -   Extract binary mask `M_c`.
        -   `M_c ← morphological_cleanup(M_c)` (Remove noise < 2000px).
        -   `labels ← connectedComponents(M_c)`.
    *   **NMS (Non-Maximum Suppression)**: 
        -   Generate boxes for each unique label.
        -   IF `IoU(Box_A, Box_B) > 0.5` → Suppress lower confidence box.
*   **RESULT**: Semantic mask + List of detected terrain objects with confidence scores.

---

## Algorithm 2: Multidimensional Suitability & Archetyping
**File**: `backend/crop_recommender.py`

This algorithm calculates the "Agricultural Potential" of a land-cover vector.

### Step-by-Step Logic:
1.  **Vector Extraction**: 
    -   `Observed ← [Urban, Agri, Barren, Forest, Range, Water]`.
2.  **Terrain Archetype Detection**:
    -   IF `Water > 15%` → Archetype = **"Wetland/Paddy Zone"**.
    -   IF `Forest > 20%` → Archetype = **"Agroforestry"**.
    -   IF `Barren > 30%` → Archetype = **"Arid Dryland"**.
3.  **Similarity Scoring (Similarity v4)**:
    -   `diff ← |Observed - Ideal_Crop_Profile|`.
    -   **Renormalization**: If `Agri > 70%`, amplify other weights: `w_other ← w_other * 1.5`.
    -   `Raw_Score ← 100 - SQRT(MEAN(diff² * weights))`.
4.  **Dynamic Bonuses**:
    -   Add `Archetype_Bonus` (+5 to +15 pts) if crop scientific profile matches detected terrain.
    -   Add `Seasonal_Bonus` (+5 pts) if current month matches planting window.
    -   Add `Rotational_Bonus` (+8 pts) if crop fixes nitrogen for previous ground cover.
*   **RESULT**: A suitability score (0-100) per crop.

---

## Algorithm 3: Maximal Marginal Relevance (MMR) Diversity
**File**: `backend/crop_recommender.py`

This algorithm ensures that the 15 recommendations are not 15 variations of the same crop (e.g., 15 cereals).

### Step-by-Step Logic:
1.  **Initialize**: `Selected ← {Highest Scored Crop}`, `Remaining ← {All other crops}`.
2.  **Iterate** (until 15 crops selected):
    -   FOR `crop` IN `Remaining`:
        -   **Relevance** = `suitability_score / 100`.
        -   **Redundancy** = `MAX(CosineSimilarity(crop.profile, Selected_Crops.profiles))`.
        -   **MMR_Score** = `0.74 * Relevance - 0.26 * Redundancy`.
        -   **Category Penalty**: IF `category_count[crop.cat] > 3` → `MMR_Score -= 0.25`.
3.  **Select**: Pick `crop` with highest `MMR_Score`.
*   **RESULT**: A diverse list balancing high scores with variety across Spices, Fruits, Cereals, etc.

---

## Algorithm 4: Logic-Driven XAI Reasoner
**File**: `backend/crop_explanations.py`

This algorithm builds a human-level agronomic argument for each recommendation.

### Step-by-Step Logic:
1.  **Dominant Driver ID**:
    -   Find land-cover type with the highest positive contribution to the score.
    -   Example: `Water` for Jute.
2.  **Narrative Selection**:
    -   Lookup crop-specific template (e.g., `explain_rice`).
    -   **Context Mapping**: 
        -   IF `Water_Observed < Ideal` → Generate "Deficit Alert" + Remediation.
        -   IF `Barren_Observed > 20%` → Generate "Salinity/Hardpan Warning".
3.  **Monte Carlo Risk Projection**:
    -   Run 100 simulations with ±2% noise in land-cover input.
    -   Calculate spread between 5th and 95th percentile.
    -   IF `spread > 25` → Label as **"Low Confidence (Ground Truth Required)"**.
*   **RESULT**: 3-4 sentences of specific ground-level advice for the farmer.
