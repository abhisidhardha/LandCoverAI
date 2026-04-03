# Crop Recommendation Engine — Technical Documentation

## Overview

The Crop Recommendation Engine is a 9-stage rule-based expert system that analyzes land-cover segmentation data to recommend suitable crops. It processes 6 land-cover classes (urban, agriculture, rangeland, forest, water, barren) and outputs ranked crop recommendations with confidence scores and rationale.

## Architecture

```
DL Model Output (landcover %)
        ↓
[Stage 1]  Raw Signal Extraction
        ↓
[Stage 2]  Composite Index Derivation
        ↓
[Stage 3]  Tier 1 — Feasibility Gate
        ↓
[Stage 4]  Tier 2 — Climate & Water Profile
        ↓
[Stage 5]  Tier 3 — Soil & Land Health Profile
        ↓
[Stage 6]  Tier 4 — Economic & Market Layer
        ↓
[Stage 7]  Tier 5 — Agroecological Suitability
        ↓
[Stage 8]  Scoring Engine + Crop Ranker
        ↓
[Stage 9]  Output: Ranked Crops + Rationale + Confidence
```

## Stage Details

### Stage 1: Raw Signal Extraction
- Normalizes land-cover percentages to sum to 100%
- Assigns categorical intensity labels:
  - Negligible: 0–5%
  - Low: 6–15%
  - Moderate: 16–35%
  - High: 36–60%
  - Dominant: 61–100%

### Stage 2: Composite Index Derivation
Derives 6 normalized indices (0–100):

1. **MAI (Moisture Availability Index)**
   - Formula: `(Water% × 1.0) + (Forest% × 0.6) + (Agriculture% × 0.3) + (Rangeland% × 0.1)`
   - Signals water/humidity availability

2. **SHI (Soil Health Index)**
   - Formula: `(Agriculture% × 1.0) + (Forest% × 0.7) − (Barren% × 1.2) − (Urban% × 0.4)`
   - Degraded/barren land reduces this significantly

3. **ASI (Aridity-Stress Index)**
   - Formula: `(Barren% × 1.0) + (Rangeland% × 0.6) − (Water% × 0.5) − (Forest% × 0.3)`
   - High ASI = dry, stressed landscape

4. **CFI (Cultivation Feasibility Index)**
   - Formula: `(Agriculture% × 1.0) + (Rangeland% × 0.5) + (Barren% × 0.2) − (Urban% × 0.5) − (Forest% × 0.2)`
   - Measures physically convertible land

5. **MEI (Market & Economic Index)**
   - Formula: `(Urban% × 1.0) + (Agriculture% × 0.3)`
   - Higher urban % = nearby market access

6. **AFI (Agroforestry Potential Index)**
   - Formula: `(Forest% × 1.0) + (Water% × 0.4) + (Rangeland% × 0.2) − (Barren% × 0.5)`
   - Signals tree-crop/shade system viability

### Stage 3: Tier 1 — Feasibility Gate
Hard gate that halts recommendation if:
- **F1**: Urban% > 65% → "Recommend rooftop/vertical farming only"
- **F2**: Barren% > 70% AND Water% < 5% AND Agriculture% < 5% → "Recommend land reclamation"
- **F3**: Water% > 60% → "Recommend aquaculture or floating agriculture"
- **F4**: CFI < 15 → "Insufficient cultivable area"
- **F5**: CFI < 30 → Pass with "Marginal land warning"

### Stage 4: Tier 2 — Climate & Water Profile
Assigns one of 5 water regimes:

1. **WATER_RICH**: MAI ≥ 65 AND Water% ≥ 20
   - Unlocks: rice, sugarcane, jute, lotus, water chestnut, taro
   - Flag: flood risk if Water% > 40

2. **HUMID**: 45 ≤ MAI < 65 AND Forest% ≥ 15
   - Unlocks: banana, coffee, cardamom, turmeric, ginger, rubber, cocoa

3. **SUB_HUMID**: 30 ≤ MAI < 45
   - Unlocks: wheat, maize, soybean, groundnut, cotton, sunflower, tomato

4. **SEMI_ARID**: 15 ≤ MAI < 30 AND ASI ≥ 30
   - Unlocks: sorghum, millet, chickpea, mustard, sesame, castor

5. **ARID**: MAI < 15 AND ASI ≥ 50
   - Unlocks: pearl millet, drought-tolerant legumes, cactus pear
   - Flag: "Low yield probability. Irrigation investment recommended"

### Stage 5: Tier 3 — Soil & Land Health Profile
Assigns soil class:

1. **PRODUCTIVE** (SHI ≥ 60)
   - No restrictions
   - Boost: wheat, rice, maize, soybean, vegetables (+15 pts)

2. **MODERATE** (35 ≤ SHI < 60)
   - Neutral scoring

3. **DEGRADED** (SHI < 35 AND Barren% ≥ 20)
   - Force-add: nitrogen-fixing crops (cowpea, pigeon pea, groundnut)
   - Penalize: rice, sugarcane, vegetables (−20 pts)

4. **RECLAMATION** (Barren% > 45 AND SHI < 20)
   - Primary recommendation: soil reclamation crops (dhaincha, sunhemp, sesbania, vetiver)
   - Note: "Revisit after 1–2 seasons"

### Stage 6: Tier 4 — Economic & Market Layer
Assigns market class:

1. **HIGH_VALUE** (MEI ≥ 55)
   - Boost: vegetables, fruits, flowers, herbs (+20 pts)
   - Rationale: perishables viable due to short supply chain

2. **COMMERCIAL** (25 ≤ MEI < 55)
   - Boost: cash crops (cotton, oilseeds, pulses, spices) (+10 pts)

3. **SUBSISTENCE** (MEI < 25)
   - Boost: staple crops (millets, sorghum, maize, pulses) (+10 pts)
   - Penalize: perishables, export crops (−15 pts)

### Stage 7: Tier 5 — Agroecological Suitability Filter
Filters crops based on:
- Required water regime match
- Minimum MAI, SHI, CFI thresholds
- Maximum ASI tolerance
- Market class compatibility

### Stage 8: Scoring Engine
Dynamic score calculation:
```
Final Score = Base Score
            + MAI match bonus        (0 to +20)
            + SHI match bonus        (0 to +15)
            + CFI bonus              (0 to +10)
            + MEI market bonus       (0 to +20)
            + AFI agroforestry bonus (0 to +10)
            − ASI stress penalty     (0 to −20)
            − Soil degradation flag  (0 to −20)
            + Regime match bonus     (+10)
            + Productive soil boost  (+15)
```

Confidence levels:
- **HIGH** (≥75): ✅ Strongly Recommended
- **MODERATE** (50–74): ⚠️ Recommended with conditions
- **LOW** (30–49): 🔶 Possible but marginal
- **NOT VIABLE** (<30): ❌ Removed from output

### Stage 9: Output Format
```json
{
  "status": "ok",
  "halt_message": null,
  "flags": ["Marginal land warning: CFI below 30."],
  "water_regime": "SEMI_ARID",
  "secondary_regime": "SUB_HUMID",
  "soil_class": "MODERATE",
  "market_class": "COMMERCIAL",
  "indices": {
    "MAI": 28.5,
    "SHI": 42.3,
    "ASI": 35.8,
    "CFI": 28.0,
    "MEI": 32.5,
    "AFI": 18.2
  },
  "signals": {
    "urban": 5.0,
    "agri": 45.0,
    "range": 15.0,
    "forest": 20.0,
    "water": 10.0,
    "barren": 5.0
  },
  "ranked_crops": [
    {
      "rank": 1,
      "crop": "Sorghum / Jowar",
      "category": "Cereal",
      "season": "Kharif",
      "score": 78.5,
      "regime_match": true,
      "marginal": false,
      "agroforestry": false
    }
  ]
}
```

## Crop Knowledge Base

The system includes 100+ crops across categories:
- **Cereals**: Rice, Wheat, Maize, Sorghum, Millets (15 varieties)
- **Pulses**: Chickpea, Lentil, Pigeon Pea, Soybean (12 varieties)
- **Oilseeds**: Groundnut, Mustard, Sunflower, Sesame (10 varieties)
- **Fiber**: Cotton, Jute, Hemp (5 varieties)
- **Sugar**: Sugarcane, Sugar Beet
- **Plantation**: Tea, Coffee, Rubber, Cocoa (7 varieties)
- **Spices**: Turmeric, Ginger, Cardamom, Pepper (14 varieties)
- **Vegetables**: Potato, Tomato, Onion, Cabbage (16 varieties)
- **Fruits**: Mango, Banana, Citrus, Grapes (17 varieties)
- **Other**: Moringa, Jatropha (reclamation crops)

## API Usage

### Endpoint: `/api/recommend-crops`

**Request:**
```json
POST /api/recommend-crops
Authorization: Bearer <token>
Content-Type: application/json

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

**Response:**
```json
{
  "status": "ok",
  "ranked_crops": [...],
  "indices": {...},
  "flags": [...]
}
```

## Integration with Prediction Pipeline

The crop recommendation engine is automatically invoked after land-cover segmentation:

1. User uploads satellite image or selects coordinates
2. Segmentation model produces class mask
3. Land-cover percentages computed from mask
4. Crop recommendation engine processes percentages
5. Results returned with both segmentation and crop recommendations

## Testing

Run the test script:
```bash
cd backend
python test_recommender.py
```

This will test 5 scenarios:
1. Balanced agricultural land
2. Arid/semi-arid region
3. Water-rich region
4. Urban dominated (halt expected)
5. Degraded soil

## Future Enhancements

1. **Climate Data Integration**: Add temperature, rainfall, elevation data
2. **Soil Testing**: Integrate pH, NPK, organic matter data
3. **Market Prices**: Real-time crop price data for economic optimization
4. **Seasonal Calendars**: Crop rotation and multi-season planning
5. **Yield Prediction**: ML-based yield forecasting per crop
6. **Risk Assessment**: Pest, disease, and climate risk scoring
7. **Irrigation Planning**: Water requirement and irrigation system recommendations
8. **Fertilizer Recommendations**: NPK and micronutrient suggestions per crop

## References

- FAO Crop Water Requirements
- USDA Plant Hardiness Zones
- Indian Council of Agricultural Research (ICAR) Guidelines
- World Agroforestry Database
