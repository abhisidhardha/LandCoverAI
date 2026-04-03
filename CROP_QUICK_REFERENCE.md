# Crop Recommendation Engine — Quick Reference

## Composite Indices (0-100 scale)

| Index | Formula | Meaning |
|-------|---------|---------|
| **MAI** | Water×1.0 + Forest×0.6 + Agri×0.3 + Range×0.1 | Moisture Availability |
| **SHI** | Agri×1.0 + Forest×0.7 - Barren×1.2 - Urban×0.4 | Soil Health |
| **ASI** | Barren×1.0 + Range×0.6 - Water×0.5 - Forest×0.3 | Aridity Stress |
| **CFI** | Agri×1.0 + Range×0.5 + Barren×0.2 - Urban×0.5 - Forest×0.2 | Cultivation Feasibility |
| **MEI** | Urban×1.0 + Agri×0.3 | Market & Economic Access |
| **AFI** | Forest×1.0 + Water×0.4 + Range×0.2 - Barren×0.5 | Agroforestry Potential |

## Water Regimes

| Regime | MAI Range | ASI | Key Crops |
|--------|-----------|-----|-----------|
| **WATER_RICH** | ≥65 | - | Rice, Sugarcane, Jute, Banana |
| **HUMID** | 45-64 | - | Coffee, Cocoa, Rubber, Turmeric |
| **SUB_HUMID** | 30-44 | - | Wheat, Maize, Cotton, Soybean |
| **SEMI_ARID** | 15-29 | ≥30 | Sorghum, Millet, Chickpea, Mustard |
| **ARID** | <15 | ≥50 | Pearl Millet, Castor, Date Palm |

## Soil Classes

| Class | SHI Range | Barren% | Action |
|-------|-----------|---------|--------|
| **PRODUCTIVE** | ≥60 | - | No restrictions, boost cereals/vegetables |
| **MODERATE** | 35-59 | - | Neutral scoring |
| **DEGRADED** | <35 | ≥20 | Force nitrogen-fixers, penalize water-intensive |
| **RECLAMATION** | <20 | >45 | Only reclamation crops (1-2 seasons) |

## Market Classes

| Class | MEI Range | Boost | Penalize |
|-------|-----------|-------|----------|
| **HIGH_VALUE** | ≥55 | Vegetables, Fruits, Herbs (+20) | - |
| **COMMERCIAL** | 25-54 | Cash crops (+10) | - |
| **SUBSISTENCE** | <25 | Staples (+10) | Perishables (-15) |

## Feasibility Gates (Halt Conditions)

| Rule | Condition | Message |
|------|-----------|---------|
| **F1** | Urban% > 65 | "Recommend rooftop/vertical farming only" |
| **F2** | Barren% > 70 AND Water% < 5 AND Agri% < 5 | "Recommend land reclamation first" |
| **F3** | Water% > 60 | "Recommend aquaculture/floating agriculture" |
| **F4** | CFI < 15 | "Insufficient cultivable area" |

## Scoring Components

| Component | Range | Condition |
|-----------|-------|-----------|
| Base Score | 40-60 | Crop-specific baseline |
| MAI Match | 0 to +20 | MAI ≥ crop.min_MAI |
| SHI Match | 0 to +15 | SHI ≥ crop.min_SHI |
| CFI Bonus | 0 to +10 | CFI ≥ crop.min_CFI |
| MEI Market | 0 to +20 | Market class match |
| AFI Agroforestry | 0 to +10 | Agroforestry compatible |
| ASI Stress | 0 to -20 | ASI > crop.max_ASI |
| Soil Degradation | 0 to -20 | Marginal soil conditions |
| Regime Match | +10 | Primary regime match |
| Productive Soil | +15 | SHI ≥ 60 for cereals/vegetables |

## Confidence Levels

| Score | Level | Symbol | Meaning |
|-------|-------|--------|---------|
| ≥75 | HIGH | ✅ | Strongly Recommended |
| 50-74 | MODERATE | ⚠️ | Recommended with conditions |
| 30-49 | LOW | 🔶 | Possible but marginal |
| <30 | NOT VIABLE | ❌ | Removed from output |

## Crop Categories (100+ crops)

- **Cereals** (15): Rice, Wheat, Maize, Sorghum, Millets
- **Pulses** (12): Chickpea, Lentil, Pigeon Pea, Soybean, Cowpea
- **Oilseeds** (10): Groundnut, Mustard, Sunflower, Sesame, Castor
- **Fiber** (5): Cotton, Jute, Hemp, Kenaf
- **Sugar** (2): Sugarcane, Sugar Beet
- **Plantation** (7): Tea, Coffee, Rubber, Cocoa, Tobacco
- **Spices** (14): Turmeric, Ginger, Cardamom, Pepper, Cumin
- **Vegetables** (16): Potato, Tomato, Onion, Cabbage, Garlic
- **Fruits** (17): Mango, Banana, Citrus, Grapes, Papaya
- **Other** (2): Moringa, Jatropha (reclamation)

## Python Usage

```python
from crop_recommender import recommend_crops

# Input: land-cover percentages
result = recommend_crops({
    "urban_land": 5,
    "agriculture": 45,
    "rangeland": 15,
    "forest": 20,
    "water": 10,
    "barren": 5
}, top_n=10)

# Output structure
{
    "status": "ok" | "halted",
    "halt_message": str | None,
    "flags": [str],
    "water_regime": str,
    "secondary_regime": str | None,
    "soil_class": str,
    "market_class": str,
    "indices": {MAI, SHI, ASI, CFI, MEI, AFI},
    "signals": {urban, agri, range, forest, water, barren},
    "ranked_crops": [
        {
            "rank": int,
            "crop": str,
            "category": str,
            "season": str,
            "score": float,
            "regime_match": bool,
            "marginal": bool,
            "agroforestry": bool
        }
    ]
}
```

## Common Scenarios

### Scenario 1: Balanced Agricultural Land
```
Input: Agri=45%, Forest=20%, Water=10%, Range=15%, Urban=5%, Barren=5%
→ SUB_HUMID regime, PRODUCTIVE soil, COMMERCIAL market
→ Recommends: Wheat, Maize, Cotton, Soybean, Groundnut
```

### Scenario 2: Arid Degraded Land
```
Input: Barren=48%, Range=35%, Agri=10%, Urban=2%, Forest=3%, Water=2%
→ ARID regime, DEGRADED soil, SUBSISTENCE market
→ Recommends: Pearl Millet, Cowpea, Cluster Bean, Castor
→ Flags: "Low yield probability. Irrigation investment recommended"
```

### Scenario 3: Water-Rich Region
```
Input: Water=35%, Forest=30%, Agri=25%, Range=5%, Urban=3%, Barren=2%
→ WATER_RICH regime, PRODUCTIVE soil, COMMERCIAL market
→ Recommends: Rice, Sugarcane, Jute, Banana, Taro
→ Flags: "Flood risk: Water% > 40"
```

### Scenario 4: Urban Dominated (Halted)
```
Input: Urban=75%, Agri=10%, Range=5%, Forest=5%, Water=3%, Barren=2%
→ Status: "halted"
→ Message: "Predominantly urban. Recommend rooftop/vertical farming only."
```

### Scenario 5: Reclamation Zone
```
Input: Barren=50%, Range=25%, Agri=15%, Urban=5%, Forest=3%, Water=2%
→ RECLAMATION soil class
→ Recommends: Moringa, Jatropha, Cowpea, Pigeon Pea (soil builders)
→ Flags: "Phase 1 — soil reclamation crops for 1-2 seasons"
```

## Testing

```bash
# Run test suite
cd backend
python test_recommender.py

# Test API endpoint
python example_crop_api.py
```

## Integration Points

1. **After Segmentation**: Extract `landcover_percentages` from mask
2. **Call Recommender**: Pass percentages to `recommend_crops()`
3. **Display Results**: Show ranked crops with confidence levels
4. **Show Rationale**: Display indices, flags, and regime/soil/market classes

## Performance Notes

- Average execution time: <50ms per recommendation
- No external API calls required
- Fully deterministic (same input → same output)
- Thread-safe (stateless function)
