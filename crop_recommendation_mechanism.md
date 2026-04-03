# LandCoverAI: Crop Recommendation Engine Mechanism

This document provides a highly detailed, step-by-step technical breakdown of how the LandCoverAI Crop Recommendation Engine operates. It covers the entire lifecycle from satellite image segmentation to the final display of Monte Carlo risk assessments and crop rotation synergies.

---

## 1. Land Cover Extraction (The Foundation)
Before any agricultural logic is applied, the system must understand the terrain.
1. A **UNet++ Segmentation Model** processes the input satellite image (either uploaded or fetched via coordinates).
2. Every pixel in the image is classified into one of 6 categories: **Urban, Agriculture, Barren, Forest, Rangeland, Water**.
3. The server aggregates these pixel counts into an **Observed Percentage Array**: `[urban%, agri%, barren%, forest%, rangeland%, water%]`.
*Example Observation: [5%, 65%, 0%, 15%, 5%, 10%]*

---

## 2. The 100-Crop FAO Knowledge Base
The system houses an array of 100 global crops (`CROP_DATA`) grounded in FAO Agro-Ecological Zones and ICAR standards. 
Every crop has a **Favorable Percentage Vector** defining its ideal environmental makeup. 
*Example - Rice (Paddy): `[2%, 88%, 2%, 4%, 6%, 82%]` (requires massive agricultural base and heavy water presence).*

---

## 3. Core Suitability Scoring (The Engine)
The heavily weighted scoring algorithm explicitly compares the **Observed Terrain** against the **target crop's Favorable Vector**.

### A. Feature Weighting
Not all land cover types exert equal influence on agricultural success. The engine applies an importance array (`WEIGHT_ARRAY`):
*   **Agriculture (0.35)**: Existing farmland is the strongest indicator of soil quality.
*   **Rangeland (0.18)**: Critical indicator for rainfed or hardy crops.
*   **Forest (0.15)**: Important for plantation, shade, or cash crops (Coffee, Cocoa).
*   **Barren (0.12)**: Aridity tolerance indicator.
*   **Water (0.12)**: Proximity necessary for hydro-intensive crops.
*   **Urban (0.08)**: Weakest indicator, typically just limits acreage.

### B. SHAP-Style Feature Contributions
For each of the 6 features, the engine computes a mathematically signed contribution:
*   **Positive Score**: If the observed land cover is abundant, and the crop *loves* that cover type, you gain massive points (e.g., matching High Agriculture).
*   **Negative Penalty**: If a land cover is highly abundant (e.g., >20% Barren) but the crop has near-zero affinity for it, the algorithm generates a strict penalty because the land is fundamentally toxic to the crop.

### C. Contextual Overrides
The base score is then subjected to specific biophysical rules:
*   **Water Boost**: If the crop demands water (>10% favorable) and water is observed, a multiplier bonus (+up to 12 pts) is added.
*   **Urban/Barren Crushing Penalty**: If Urban > 50% or Barren > 40%, severe absolute penalties are applied.

---

## 4. Advanced System Dynamics (The "Intelligence" Layer)
This is where the engine transforms from a basic math script into a living agronomical intelligence tool.

### A. Temporal Seasonal Suitability
The system dynamically calculates the current month (`datetime.now().month`) and references `SEASONAL_DATA`.
1. It cross-checks the month against India's three major planting windows: **Kharif (June-Sept)**, **Rabi (Oct-Mar)**, and **Zaid (Apr-June)**.
2. If a crop natively thrives in the *current* active season, the engine applies an absolute **+10 Suitability Bonus**.
3. *Why?* To ensure that the top recommendations presented to the user are actually viable to plant right now, rather than theoretical recommendations for 6 months from now.

### B. Rotational Synergy (The "Follow-Up / Previous Crop" Logic)
If the user selects a "Previous Crop" in the UI (e.g., "I just harvested Soybean"), the payload triggers the `ROTATION_RULES` engine. Crop succession is vital for soil health and pest management.

*   **Synergy Bonus (+15 Points)**: The engine searches for natural successors. For example, if the previous crop was Soybean (a Legume), it fixed immense amounts of nitrogen into the topsoil. The engine will heavily boost subsequent Cereals (like Wheat or Maize) because they perfectly exploit that nitrogen. The UI will render a green badge explaining this synergy.
*   **Lockout Penalty (-30 Points)**: To prevent disease cycling and soil depletion, the engine penalizes consecutive planting of the same crop. If you just grew Rice, the engine subtracts 30 points from Rice, virtually guaranteeing it falls off the top 10 list, forcing a balanced rotation.

### C. Monte Carlo Risk Assessment
To guarantee confidence, the suite runs a **100-iteration Monte Carlo simulation**.
1. The engine blasts the observed land cover percentages with Gaussian noise (±5%), simulating real-world sensor inaccuracies and edge-case geography.
2. It scores the crop 100 times under this chaotic noise.
3. This generates a **90% Confidence Interval** (e.g., 82% - 94%).
4. **Prediction Risk Badge**: If the spread is narrow (<10 point variance), the crop gets a green **Low Risk** badge. If the terrain is highly sensitive to minor deviations, it gets a **Moderate** or **High Risk** badge.

### D. Counterfactual Optimizations
If a crop scores moderately (e.g., 65%), the engine calculates the exact mathematical gradients needed to make the crop viable. 
- It asks: *"What is the single largest deficit harming this crop?"*
- It generates actionable advice: *"Increase Agriculture by +20%"* or *"Reduce Barren by -15%"*.

---

## 5. Narrative Building & Payload Generation
Finally, the system packages all this data into a coherent payload:
1.  **Dynamic Explanations (`crop_explanations.py`)**: The `build_explanation` function acts like an LLM. It ingests the exact land cover observations and hard-codes sentences specifically justifying the score. E.g., *"Agricultural land at 65% is an excellent foundation for soybean..."*
2.  **Matplotlib SHAP Charts**: A headless Python plotter (`generate_visual_explanation`) takes the 6 feature contributions and generates a beautiful base64-encoded horizontal bar chart, proving visually exactly why the crop scored the way it did.

## 6. Frontend Interpretation
The backend trims the array to the **Top 10** crops and ships it via JSON. The browser (`predict.js`) unpacks the JSON and constructs the rich UI cards containing the Risk Badges, Rotation Synergy warnings, Plant Windows, and Matplotlib visual analytics over the map.
