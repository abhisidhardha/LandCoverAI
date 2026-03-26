"""
Crop Recommendation Engine — FAO/ICAR/CGIAR Suitability Data
==============================================================
100 global crops with land cover suitability percentages sourced from
FAO Agro-Ecological Zones, ICAR crop guidelines, and CGIAR suitability maps.

Scoring: converts segmentation mask pixels → land cover %, then computes
a weighted suitability score by comparing observed % against each crop's
favorable % per land cover class. Provides per-feature contribution
explanations (SHAP-style analytic decomposition).
"""

import logging
import numpy as np
from typing import List, Dict, Optional
from crop_explanations import build_explanation

# ==============================================================================
# CROP DATABASE — 100 Crops (FAO/ICAR/CGIAR Suitability Percentages)
# ==============================================================================
# Each entry: [id, name, scientific_name, category,
#              urban%, agriculture%, barren%, forest%, rangeland%, water%]
#
# Values represent % of that land cover class globally that presents
# FAVORABLE biophysical conditions (soil, climate, water) for the crop.

CROP_DATA = [
    # ── CEREALS & MILLETS (15) ───────────────────────────────────────────────
    [1,  "Rice (Paddy)",            "Oryza sativa",                 "Cereal",      2,  88,   2,   4,   6,  82],
    [2,  "Wheat",                   "Triticum aestivum",            "Cereal",      2,  85,  18,   3,  28,   1],
    [3,  "Maize / Corn",            "Zea mays",                     "Cereal",      3,  80,   8,  10,  22,   1],
    [4,  "Sorghum / Jowar",         "Sorghum bicolor",              "Cereal",      2,  72,  35,   5,  45,   1],
    [5,  "Pearl Millet / Bajra",    "Pennisetum glaucum",           "Cereal",      1,  68,  42,   3,  52,   0],
    [6,  "Finger Millet / Ragi",    "Eleusine coracana",            "Cereal",      1,  70,  20,  15,  35,   1],
    [7,  "Barley",                  "Hordeum vulgare",              "Cereal",      2,  75,  28,   5,  38,   1],
    [8,  "Oats",                    "Avena sativa",                 "Cereal",      2,  72,  15,   8,  32,   1],
    [9,  "Rye",                     "Secale cereale",               "Cereal",      2,  68,  22,  10,  35,   1],
    [10, "Triticale",               "×Triticosecale",               "Cereal",      2,  70,  20,   8,  30,   1],
    [11, "Teff",                    "Eragrostis tef",               "Cereal",      1,  65,  30,   5,  45,   5],
    [12, "Foxtail Millet / Kangni", "Setaria italica",              "Cereal",      1,  66,  28,   6,  40,   1],
    [13, "Kodo Millet",             "Paspalum scrobiculatum",       "Cereal",      1,  62,  22,  12,  38,   3],
    [14, "Little Millet / Kutki",   "Panicum sumatrense",           "Cereal",      1,  60,  25,  14,  38,   2],
    [15, "Proso Millet",            "Panicum miliaceum",            "Cereal",      1,  64,  32,   5,  42,   0],

    # ── PULSES / LEGUMES (12) ────────────────────────────────────────────────
    [16, "Chickpea / Chana",        "Cicer arietinum",              "Pulse",       1,  75,  30,   3,  38,   0],
    [17, "Pigeon Pea / Tur (Arhar)","Cajanus cajan",                "Pulse",       1,  72,  22,   8,  30,   1],
    [18, "Lentil / Masur",          "Lens culinaris",               "Pulse",       1,  74,  20,   5,  28,   0],
    [19, "Green Gram / Moong",      "Vigna radiata",                "Pulse",       2,  73,  18,   6,  25,   1],
    [20, "Black Gram / Urad",       "Vigna mungo",                  "Pulse",       2,  72,  15,   8,  22,   1],
    [21, "Kidney Bean / Rajma",     "Phaseolus vulgaris",           "Pulse",       2,  70,  10,  12,  20,   0],
    [22, "Cowpea / Lobia",          "Vigna unguiculata",            "Pulse",       2,  68,  35,   5,  42,   0],
    [23, "Field Pea",               "Pisum sativum",                "Pulse",       2,  72,  12,   8,  25,   0],
    [24, "Soybean",                 "Glycine max",                  "Pulse",       2,  80,   8,  10,  18,   1],
    [25, "Moth Bean",               "Vigna aconitifolia",           "Pulse",       1,  60,  48,   2,  55,   0],
    [26, "Horse Gram / Kulthi",     "Macrotyloma uniflorum",        "Pulse",       1,  62,  32,  10,  40,   0],
    [27, "Cluster Bean / Guar",     "Cyamopsis tetragonoloba",      "Pulse",       1,  64,  40,   2,  48,   0],

    # ── OILSEEDS (10) ────────────────────────────────────────────────────────
    [28, "Groundnut / Peanut",      "Arachis hypogaea",             "Oilseed",     2,  78,  20,   5,  25,   0],
    [29, "Mustard / Rapeseed",      "Brassica juncea / napus",      "Oilseed",     2,  80,  18,   4,  25,   1],
    [30, "Sunflower",               "Helianthus annuus",            "Oilseed",     2,  75,  20,   3,  30,   1],
    [31, "Sesame / Til",            "Sesamum indicum",              "Oilseed",     1,  70,  28,   5,  35,   0],
    [32, "Linseed / Flax",          "Linum usitatissimum",          "Oilseed",     2,  72,  18,   5,  28,   1],
    [33, "Castor",                  "Ricinus communis",             "Oilseed",     2,  65,  38,   5,  42,   0],
    [34, "Safflower",               "Carthamus tinctorius",         "Oilseed",     1,  65,  35,   3,  40,   0],
    [35, "Niger / Ramtil",          "Guizotia abyssinica",          "Oilseed",     1,  62,  25,  15,  35,   2],
    [36, "Palm Oil",                "Elaeis guineensis",            "Oilseed",     1,  82,   2,  55,   8,   2],
    [37, "Coconut",                 "Cocos nucifera",               "Oilseed",     5,  75,   5,  30,  10,   8],

    # ── FIBER CROPS (5) ──────────────────────────────────────────────────────
    [38, "Cotton (American Upland)","Gossypium hirsutum",           "Fiber",       2,  80,  15,   3,  20,   0],
    [39, "Cotton (Desi)",           "Gossypium arboreum",           "Fiber",       2,  72,  20,   4,  22,   0],
    [40, "Jute",                    "Corchorus olitorius/capsularis","Fiber",       1,  82,   3,   8,   5,  15],
    [41, "Hemp",                    "Cannabis sativa",              "Fiber",       3,  70,  10,  15,  25,   2],
    [42, "Kenaf",                   "Hibiscus cannabinus",          "Fiber",       1,  70,  12,  10,  18,   3],

    # ── SUGAR CROPS (2) ──────────────────────────────────────────────────────
    [43, "Sugarcane",               "Saccharum officinarum",        "Sugar",       2,  85,   5,   5,   8,   5],
    [44, "Sugar Beet",              "Beta vulgaris subsp. vulgaris","Sugar",       2,  80,  12,   3,  18,   1],

    # ── PLANTATION / CASH CROPS (7) ──────────────────────────────────────────
    [45, "Tea",                     "Camellia sinensis",            "Plantation",  2,  70,   2,  65,   8,   2],
    [46, "Coffee Arabica",          "Coffea arabica",               "Plantation",  2,  68,   1,  72,   5,   1],
    [47, "Coffee Robusta",          "Coffea canephora",             "Plantation",  2,  72,   1,  68,   5,   2],
    [48, "Rubber",                  "Hevea brasiliensis",           "Plantation",  1,  75,   1,  70,   3,   2],
    [49, "Tobacco",                 "Nicotiana tabacum",            "Plantation",  2,  75,  12,   5,  15,   1],
    [50, "Cocoa",                   "Theobroma cacao",              "Plantation",  1,  70,   0,  75,   2,   2],
    [51, "Cashew",                  "Anacardium occidentale",       "Plantation",  3,  68,  22,  20,  15,   1],

    # ── SPICES & CONDIMENTS (14) ─────────────────────────────────────────────
    [52, "Black Pepper",            "Piper nigrum",                 "Spice",       2,  65,   0,  70,   3,   2],
    [53, "Cardamom (Large)",        "Amomum subulatum",             "Spice",       1,  60,   0,  78,   2,   3],
    [54, "Cardamom (Small/Green)",  "Elettaria cardamomum",         "Spice",       1,  62,   0,  75,   2,   3],
    [55, "Turmeric",                "Curcuma longa",                "Spice",       2,  72,   3,  30,   5,   5],
    [56, "Ginger",                  "Zingiber officinale",          "Spice",       2,  70,   2,  35,   5,   3],
    [57, "Chili / Capsicum",        "Capsicum annuum",              "Spice",       3,  75,  10,   5,  12,   1],
    [58, "Coriander / Dhania",      "Coriandrum sativum",           "Spice",       2,  73,  18,   3,  20,   0],
    [59, "Cumin / Jeera",           "Cuminum cyminum",              "Spice",       1,  65,  35,   2,  40,   0],
    [60, "Fenugreek / Methi",       "Trigonella foenum-graecum",    "Spice",       2,  70,  22,   3,  28,   0],
    [61, "Clove",                   "Syzygium aromaticum",          "Spice",       1,  62,   0,  75,   2,   3],
    [62, "Nutmeg",                  "Myristica fragrans",           "Spice",       1,  60,   0,  72,   2,   3],
    [63, "Vanilla",                 "Vanilla planifolia",           "Spice",       1,  60,   0,  75,   1,   3],
    [64, "Saffron",                 "Crocus sativus",               "Spice",       1,  60,  35,   5,  42,   0],
    [65, "Fennel / Saunf",          "Foeniculum vulgare",           "Spice",       2,  68,  25,   5,  30,   0],

    # ── VEGETABLES & TUBERS (16) ─────────────────────────────────────────────
    [66, "Potato",                  "Solanum tuberosum",            "Vegetable",   5,  80,   8,   5,  12,   1],
    [67, "Onion",                   "Allium cepa",                  "Vegetable",   3,  78,  12,   2,  15,   1],
    [68, "Tomato",                  "Lycopersicum esculentum",      "Vegetable",   5,  78,   8,   3,  12,   1],
    [69, "Brinjal / Eggplant",      "Solanum melongena",            "Vegetable",   4,  75,   8,   5,  12,   1],
    [70, "Cabbage",                 "Brassica oleracea var. capitata","Vegetable",  4,  75,   5,   5,  10,   1],
    [71, "Cauliflower",             "Brassica oleracea var. botrytis","Vegetable",  4,  74,   5,   5,  10,   1],
    [72, "Garlic",                  "Allium sativum",               "Vegetable",   3,  76,  15,   3,  18,   0],
    [73, "Okra / Lady's Finger",    "Abelmoschus esculentus",       "Vegetable",   3,  73,  12,   5,  15,   1],
    [74, "Pumpkin / Kaddu",         "Cucurbita maxima",             "Vegetable",   3,  72,  10,   8,  18,   2],
    [75, "Cucumber",                "Cucumis sativus",              "Vegetable",   4,  73,   5,   6,  10,   2],
    [76, "Cassava / Tapioca",       "Manihot esculenta",            "Vegetable",   2,  72,  25,  20,  30,   1],
    [77, "Sweet Potato",            "Ipomoea batatas",              "Vegetable",   3,  72,  15,  10,  20,   2],
    [78, "Yam",                     "Dioscorea spp.",               "Vegetable",   2,  70,   5,  35,  10,   3],
    [79, "Taro / Colocasia",        "Colocasia esculenta",          "Vegetable",   2,  68,   2,  30,   5,  25],
    [80, "Spinach",                 "Spinacia oleracea",            "Vegetable",   5,  72,   5,   5,  10,   1],
    [81, "Carrot",                  "Daucus carota",                "Vegetable",   4,  74,   8,   5,  12,   0],

    # ── FRUITS (17) ──────────────────────────────────────────────────────────
    [82, "Mango",                   "Mangifera indica",             "Fruit",       5,  72,  15,  12,  18,   1],
    [83, "Banana",                  "Musa spp.",                    "Fruit",       4,  75,   3,  20,   8,   5],
    [84, "Papaya",                  "Carica papaya",                "Fruit",       5,  70,   8,  15,  10,   2],
    [85, "Guava",                   "Psidium guajava",              "Fruit",       6,  68,  20,  12,  25,   1],
    [86, "Pomegranate / Anar",      "Punica granatum",              "Fruit",       5,  65,  32,   5,  35,   0],
    [87, "Grapes",                  "Vitis vinifera",               "Fruit",       4,  72,  25,   5,  20,   1],
    [88, "Apple",                   "Malus domestica",              "Fruit",       3,  70,   8,  25,  15,   1],
    [89, "Orange / Citrus",         "Citrus sinensis",              "Fruit",       4,  72,  12,  10,  15,   1],
    [90, "Lemon / Lime",            "Citrus limon / aurantifolia",  "Fruit",       5,  70,  18,   8,  20,   1],
    [91, "Sapota / Chiku",          "Manilkara zapota",             "Fruit",       4,  68,  15,  15,  12,   2],
    [92, "Pineapple",               "Ananas comosus",               "Fruit",       2,  72,   5,  30,   8,   2],
    [93, "Avocado",                 "Persea americana",             "Fruit",       3,  68,   5,  35,   8,   2],
    [94, "Strawberry",              "Fragaria × ananassa",          "Fruit",       8,  70,   3,   8,  12,   1],
    [95, "Olive",                   "Olea europaea",                "Fruit",       3,  65,  38,   5,  30,   0],
    [96, "Date Palm",               "Phoenix dactylifera",          "Fruit",       3,  60,  55,   0,  25,   5],
    [97, "Jackfruit / Kathal",      "Artocarpus heterophyllus",     "Fruit",       3,  68,   2,  40,   5,   3],
    [98, "Litchi",                  "Litchi chinensis",             "Fruit",       3,  70,   3,  20,   8,   3],

    # ── OTHER CROPS (2) ──────────────────────────────────────────────────────
    [99,  "Moringa / Drumstick",    "Moringa oleifera",             "Other",       4,  65,  38,  10,  40,   1],
    [100, "Jatropha (Biofuel)",     "Jatropha curcas",              "Other",       2,  55,  45,   8,  50,   0],
]

NUM_CROPS = len(CROP_DATA)

# Segmentation model class order (0-6)
# Our model outputs: 0=urban, 1=agriculture, 2=rangeland, 3=forest, 4=water, 5=barren, 6=unknown
# Reference data columns: urban, agriculture, barren, forest, rangeland, water
# Mapping from reference column index to segmentation class ID:
#   ref[0]=urban    → seg class 0
#   ref[1]=agri     → seg class 1
#   ref[2]=barren   → seg class 5
#   ref[3]=forest   → seg class 3
#   ref[4]=rangeland→ seg class 2
#   ref[5]=water    → seg class 4

# Land cover names in DISPLAY order matching reference columns
LANDCOVER_DISPLAY = [
    ("urban_land",   0),   # seg class 0
    ("agriculture",  1),   # seg class 1
    ("barren",       5),   # seg class 5
    ("forest",       3),   # seg class 3
    ("rangeland",    2),   # seg class 2
    ("water",        4),   # seg class 4
]

# For the profile output (seg class order)
LANDCOVER_NAMES = [
    "urban_land",   # 0
    "agriculture",  # 1
    "rangeland",    # 2
    "forest",       # 3
    "water",        # 4
    "barren",       # 5
    "unknown",      # 6
]

# Feature names for explanations (matching reference column order)
FEATURE_NAMES = [
    "pct_urban",
    "pct_agriculture",
    "pct_barren",
    "pct_forest",
    "pct_rangeland",
    "pct_water",
]

# Importance weights for each land cover type in the suitability calculation
# Agriculture is the most important indicator, followed by rangeland and water
WEIGHTS = {
    "urban":     0.08,   # urban areas rarely determine crop choice
    "agriculture": 0.35, # existing farmland is strongest signal
    "barren":    0.12,   # barren tolerance matters for arid crops
    "forest":    0.15,   # forest cover matters for plantation/shade crops
    "rangeland": 0.18,   # rangeland suitability for rainfed crops
    "water":     0.12,   # water proximity critical for paddy, jute, etc.
}

WEIGHT_ARRAY = np.array([
    WEIGHTS["urban"],
    WEIGHTS["agriculture"],
    WEIGHTS["barren"],
    WEIGHTS["forest"],
    WEIGHTS["rangeland"],
    WEIGHTS["water"],
], dtype=np.float64)


# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================
def extract_landcover_percentages(class_mask: np.ndarray) -> dict:
    """
    Convert segmentation mask (H×W, values 0-6) into land cover percentages.
    
    Returns dict with:
      - pct: array of 6 percentages in reference column order
             [urban%, agri%, barren%, forest%, rangeland%, water%]
      - seg_pcts: array of 7 percentages in segmentation class order (0-6)
    """
    total = max(class_mask.size, 1)
    
    # Count each segmentation class (0-6)
    seg_pcts = np.zeros(7, dtype=np.float64)
    for cid in range(7):
        seg_pcts[cid] = float(np.sum(class_mask == cid)) / total * 100.0
    
    # Map to reference column order: [urban, agri, barren, forest, rangeland, water]
    ref_pcts = np.array([
        seg_pcts[0],  # urban
        seg_pcts[1],  # agriculture
        seg_pcts[5],  # barren
        seg_pcts[3],  # forest
        seg_pcts[2],  # rangeland
        seg_pcts[4],  # water
    ], dtype=np.float64)
    
    return {"pct": ref_pcts, "seg_pcts": seg_pcts}


# ==============================================================================
# SUITABILITY SCORING
# ==============================================================================
def _compute_suitability(observed_pct: np.ndarray, crop_row: list) -> tuple:
    """
    Score how suitable a crop is for the observed land cover distribution.
    
    For each land cover type:
      - crop's favorable% = max possible suitability on that class
      - observed% of that class in the area
      - contribution = weight × min(observed, favorable) / max(favorable, 1)
        (how much of the favorable range is covered by what's observed)
    
    Also applies bonuses/penalties:
      - If observed agriculture > 0 and crop has high agri affinity → bonus
      - If area is heavily barren but crop has low barren tolerance → penalty
      - If area has significant water & crop needs water → bonus
    
    Returns:
        (score_0_to_100, contributions_list)
    """
    # Crop's favorable percentages: [urban, agri, barren, forest, range, water]
    fav = np.array([crop_row[4], crop_row[5], crop_row[6],
                     crop_row[7], crop_row[8], crop_row[9]], dtype=np.float64)
    obs = observed_pct
    
    # Per-feature contribution: how well does observed match favorable
    contributions = np.zeros(6, dtype=np.float64)
    
    for i in range(6):
        f = max(fav[i], 0.5)  # avoid division by zero, minimum 0.5%
        o = obs[i]
        
        if fav[i] >= 1:
            # This land cover type matters for this crop
            # Score based on how much of the favorable range is present
            match_ratio = min(o / f, 1.5)  # cap at 150% (diminishing returns beyond ideal)
            
            # Apply sigmoid-like scaling: gentle near 0, steep in middle, plateaus near 1
            if match_ratio <= 1.0:
                scaled = match_ratio ** 0.7  # sub-linear — partial presence is still useful
            else:
                scaled = 1.0 + 0.1 * (match_ratio - 1.0)  # small bonus for excess
            
            contributions[i] = WEIGHT_ARRAY[i] * scaled * fav[i]
        else:
            # Crop has near-zero affinity for this class
            # Penalize if this class dominates the area — it means unsuitable land
            if o > 20:
                contributions[i] = -WEIGHT_ARRAY[i] * (o / 100.0) * 15
            else:
                contributions[i] = WEIGHT_ARRAY[i] * 0.5  # neutral small contribution
    
    # Raw score from contributions
    raw_score = float(np.sum(contributions))
    
    # Normalize: max possible = sum(weight_i × fav_i) for all features
    max_possible = float(np.sum(WEIGHT_ARRAY * np.maximum(fav, 0.5)))
    
    # Base suitability (0-100)
    if max_possible > 0:
        base = (raw_score / max_possible) * 100.0
    else:
        base = 0.0
    
    # ── Contextual bonuses & penalties ──
    bonus = 0.0
    
    # Bonus: if area has agriculture and crop is agriculture-loving
    if obs[1] > 5 and fav[1] >= 60:  # >5% agri observed, crop wants ≥60% agri
        bonus += min(15, obs[1] * 0.2)
    
    # Bonus: water-dependent crops get boost if water present
    if fav[5] >= 10 and obs[5] > 2:  # crop wants water, water present
        bonus += min(12, obs[5] * fav[5] / 20.0)
    
    # Penalty: heavily urban area & crop has very low urban tolerance
    if obs[0] > 50 and fav[0] <= 2:
        bonus -= (obs[0] - 50) * 0.3
    
    # Penalty: heavily barren area & crop can't handle barren
    if obs[2] > 40 and fav[2] <= 5:
        bonus -= (obs[2] - 40) * 0.25
    
    # Bonus: forest crops in forested areas
    if fav[3] >= 30 and obs[3] > 10:
        bonus += min(10, obs[3] * 0.2)
    
    # Bonus: rangeland crops in rangeland areas
    if fav[4] >= 25 and obs[4] > 10:
        bonus += min(8, obs[4] * 0.15)
    
    final_score = np.clip(base + bonus, 0.0, 100.0)
    
    # Build signed contribution list for explanations
    contrib_list = []
    for i, fname in enumerate(FEATURE_NAMES):
        contrib_list.append({
            "feature":    fname,
            "value":      round(float(obs[i]), 1),
            "shap_value": round(float(contributions[i]), 3),
        })
    
    # Sort by absolute contribution (descending)
    contrib_list.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    
    return round(float(final_score), 1), contrib_list


# ==============================================================================
# RECOMMENDER CLASS
# ==============================================================================
class CropRecommender:
    """
    Data-driven crop suitability scorer using FAO/ICAR/CGIAR reference data.
    Computes suitability by matching observed land cover % against each crop's
    favorable % per land cover class, with weighted importance scoring.
    """

    def __init__(self):
        # Load enriched data from JSON (if available)
        self._enriched = {}
        try:
            import json, os
            json_path = os.path.join(os.path.dirname(__file__), "crop_suitability_data.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for crop in data.get("crops", []):
                    self._enriched[crop["id"]] = crop
                logging.info(f"Loaded enriched data for {len(self._enriched)} crops")
        except Exception as e:
            logging.warning(f"Could not load enriched crop data: {e}")

        logging.info(f"CropRecommender initialized with {NUM_CROPS} crops "
                     f"(FAO/ICAR/CGIAR data, no training needed)")

    def recommend_and_explain(self, observed_pct: np.ndarray, top_k: int = 10) -> tuple:
        """
        Score all 100 crops and return top-K with explanations.
        
        Args:
            observed_pct: array of 6 percentages [urban, agri, barren, forest, range, water]
            top_k: number of results to return
        
        Returns:
            (recommendations_list, explanations_dict)
        """
        all_results = []
        all_explanations = {}

        for crop in CROP_DATA:
            score, contribs = _compute_suitability(observed_pct, crop)
            
            rec = {
                "crop_id":           crop[0],
                "name":              crop[1],
                "scientific_name":   crop[2],
                "category":          crop[3],
                "suitability_score": score,
                "favorable": {
                    "urban":      crop[4],
                    "agriculture": crop[5],
                    "barren":     crop[6],
                    "forest":     crop[7],
                    "rangeland":  crop[8],
                    "water":      crop[9],
                },
            }

            # Merge enriched agricultural data
            enriched = self._enriched.get(crop[0])
            if enriched:
                gc = enriched.get("growing_conditions", {})
                rec["growing_conditions"] = gc
                rec["fertilizers"] = enriched.get("fertilizers", "")
                rec["best_regions"] = enriched.get("best_regions_india", "")
                rec["key_practices"] = enriched.get("key_practices", "")
                
            # Build explanation from observed land cover — no static text dependency
            fav_pct = [crop[4], crop[5], crop[6], crop[7], crop[8], crop[9]]
            explanation = build_explanation(
                crop_id  = crop[0],
                obs      = observed_pct,
                fav      = fav_pct,
                score    = score,
                contribs = contribs,
            )
            
            # Use dynamic if available, fallback to static Agronomic Profile
            if explanation:
                rec["explanation"] = f"<strong>AI Land Analysis:</strong> {explanation}"
            elif enriched and enriched.get("explanation"):
                rec["explanation"] = f"<strong>Agronomic Profile:</strong> {enriched.get('explanation')}"

            all_results.append(rec)
            all_explanations[crop[0]] = contribs

        # Sort by suitability descending
        all_results.sort(key=lambda r: r["suitability_score"], reverse=True)

        top = all_results[:top_k]
        top_expl = {str(r["crop_id"]): all_explanations[r["crop_id"]] for r in top}

        return top, top_expl


# ==============================================================================
# SINGLETON
# ==============================================================================
_recommender_instance: Optional[CropRecommender] = None


def get_recommender() -> CropRecommender:
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = CropRecommender()
    return _recommender_instance


# ==============================================================================
# HIGH-LEVEL API
# ==============================================================================
def generate_recommendations(class_mask: np.ndarray, top_k: int = 20) -> Dict:
    """
    Full recommendation pipeline:
      1. Convert segmentation mask pixels → land cover percentages
      2. Score all 100 crops against observed land cover
      3. Return top-K recommendations with feature contributions

    Args:
        class_mask: 2D int array (H, W) with class IDs 0-6
        top_k: number of recommendations

    Returns:
        Dict with keys:
          - recommendations: list of crop dicts with scores
          - explanations: dict mapping crop_id → per-feature contributions
          - landcover_profile: dict of land cover % for display
    """
    recommender = get_recommender()

    # Extract percentages
    features = extract_landcover_percentages(class_mask)
    observed = features["pct"]
    seg_pcts = features["seg_pcts"]

    # Score & explain
    recommendations, explanations = recommender.recommend_and_explain(observed, top_k=top_k)

    # Build profile for UI (segmentation class order for display)
    profile = {}
    for cid, name in enumerate(LANDCOVER_NAMES):
        profile[name] = round(float(seg_pcts[cid]), 1)

    return {
        "recommendations":   recommendations,
        "explanations":      explanations,
        "landcover_profile": profile,
    }
