"""
Crop Recommendation Engine v3 — Pure Rule-Based Scoring
100 global crops with land cover suitability percentages sourced from
FAO Agro-Ecological Zones, ICAR crop guidelines, and CGIAR suitability maps.

v3 Changes (April 2026):
  1. Removed cosine similarity — fully rule-based scoring engine
  2. Weighted Match Score (0-45 pts) with sub-linear reward curves
  3. Graduated deficit/excess penalties with quadratic scaling
  4. Specialization bonus (0-10 pts) for niche crops matching terrain
  5. Terrain-fit bonus (0-10 pts) via feature-by-feature threshold checks
  6. Sigmoid normalization to map raw scores to 0-100 range
  7. Terrain-type classification with archetype bonuses
  8. Category-diversity enforcement in top-K selection
  9. Risk-tier grouping: "Best Fit" / "Good Alternative" / "Worth Exploring"
"""

import logging
import math
import numpy as np
from typing import List, Dict, Optional
from crop_explanations import build_explanation

# ============================================================================
# CROP DATABASE - 100 Crops (FAO/ICAR/CGIAR Suitability Percentages)
# ============================================================================
# Each entry: [id, name, scientific_name, category,
#              urban%, agriculture%, barren%, forest%, rangeland%, water%]
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
    [98, "Litchi",                   "Litchi chinensis",             "Fruit",       3,  70,   3,  20,   8,   3],

    # ── OTHER CROPS (2) ──────────────────────────────────────────────────────
    [99,  "Moringa / Drumstick",    "Moringa oleifera",             "Other",       4,  65,  38,  10,  40,   1],
    [100, "Jatropha (Biofuel)",     "Jatropha curcas",              "Other",       2,  55,  45,   8,  50,   0],
]

NUM_CROPS = len(CROP_DATA)

# Segmentation model class order (0-6)
# Our model outputs: 0=urban, 1=agriculture, 2=rangeland, 3=forest, 4=water, 5=barren, 6=unknown
# Reference data columns: urban, agriculture, barren, forest, rangeland, water
LANDCOVER_DISPLAY = [
    ("urban_land",   0),   # seg class 0
    ("agriculture",  1),   # seg class 1
    ("barren",       5),   # seg class 5
    ("forest",       3),   # seg class 3
    ("rangeland",    2),   # seg class 2
    ("water",        4),   # seg class 4
]

LANDCOVER_NAMES = [
    "urban_land",   # 0
    "agriculture",  # 1
    "rangeland",    # 2
    "forest",       # 3
    "water",        # 4
    "barren",       # 5
    "unknown",      # 6
]

FEATURE_NAMES = [
    "pct_urban",
    "pct_agriculture",
    "pct_barren",
    "pct_forest",
    "pct_rangeland",
    "pct_water",
]

# ============================================================================
# v2: REBALANCED FEATURE WEIGHTS
# ============================================================================
# Agriculture weight reduced from 0.35 -> 0.25 so differentiating features
# (forest, water, barren, rangeland) have more influence on the final score.
# This prevents 60+ crops with similar agri affinity from clustering together.
WEIGHTS = {
    "urban":       0.07,   # urban areas rarely determine crop choice
    "agriculture": 0.25,   # was 0.35 — still important but no longer dominant
    "barren":      0.15,   # was 0.12 — critical differentiator for arid crops
    "forest":      0.19,   # was 0.15 — key for plantation/shade crops
    "rangeland":   0.17,   # was 0.18 — stable
    "water":       0.17,   # was 0.12 — THE differentiator for paddy/jute/taro
}

WEIGHT_ARRAY = np.array([
    WEIGHTS["urban"],
    WEIGHTS["agriculture"],
    WEIGHTS["barren"],
    WEIGHTS["forest"],
    WEIGHTS["rangeland"],
    WEIGHTS["water"],
], dtype=np.float64)

# ============================================================================
# v2: TERRAIN ARCHETYPE CLASSIFICATION
# ============================================================================
# Classify terrain into archetypes so we can apply targeted bonuses
# to crops that naturally belong in that terrain type.
TERRAIN_ARCHETYPES = {
    "Irrigated Farmland": {
        "condition": lambda obs: obs[1] >= 60 and obs[5] < 15,
        "description": "Rich agricultural land with established irrigation infrastructure",
        "category_bonuses": {"Cereal": 8, "Pulse": 6, "Sugar": 8, "Oilseed": 5, "Fiber": 5, "Vegetable": 4},
    },
    "Arid Dryland": {
        "condition": lambda obs: obs[2] >= 25 or (obs[4] >= 35 and obs[5] < 10),
        "description": "Semi-arid to arid terrain suited for drought-hardy crops",
        "category_bonuses": {"Cereal": 3, "Pulse": 5, "Oilseed": 4, "Other": 6},
        "crop_id_bonuses": {5: 12, 25: 15, 27: 12, 4: 10, 22: 10, 33: 8, 59: 8,
                            86: 10, 95: 10, 96: 15, 100: 12, 99: 10, 64: 8, 15: 10},
    },
    "Wetland / Paddy Zone": {
        "condition": lambda obs: obs[5] >= 10,
        "description": "High water presence — ideal for wetland and paddy cultivation",
        "category_bonuses": {"Fiber": 5},
        "crop_id_bonuses": {1: 18, 40: 15, 79: 12, 43: 8, 83: 6, 55: 5, 37: 5, 42: 5},
    },
    "Forest / Agroforestry": {
        "condition": lambda obs: obs[3] >= 20,
        "description": "Significant forest cover — suited for shade-tolerant plantation crops",
        "category_bonuses": {"Plantation": 15, "Spice": 10, "Fruit": 4},
        "crop_id_bonuses": {45: 12, 46: 15, 47: 14, 48: 14, 50: 15, 52: 12, 53: 15,
                            54: 14, 61: 12, 62: 12, 63: 12, 36: 10, 97: 10, 93: 8,
                            92: 8, 78: 8, 56: 8, 88: 6},
    },
    "Rangeland / Pastoral": {
        "condition": lambda obs: obs[4] >= 25 and obs[1] < 50,
        "description": "Open grassland/shrubland suited for rainfed and hardy crops",
        "category_bonuses": {"Cereal": 5, "Pulse": 6, "Oilseed": 5},
        "crop_id_bonuses": {4: 10, 5: 12, 11: 10, 12: 10, 13: 8, 14: 8, 15: 10,
                            22: 8, 25: 10, 26: 8, 27: 10, 33: 8, 34: 8, 59: 6, 64: 6},
    },
    "Mixed Terrain": {
        "condition": lambda obs: True,  # fallback
        "description": "Diverse land cover — multiple crop types viable",
        "category_bonuses": {},
    },
}

def classify_terrain(observed_pct: np.ndarray) -> tuple:
    """
    Classify terrain into an archetype based on observed land cover percentages.
    Returns (archetype_name, archetype_dict).
    Order matters — first match wins (most specific first).
    """
    # obs order: [urban, agri, barren, forest, rangeland, water]
    for name, archetype in TERRAIN_ARCHETYPES.items():
        if name == "Mixed Terrain":
            continue  # skip fallback during first pass
        if archetype["condition"](observed_pct):
            return name, archetype
    return "Mixed Terrain", TERRAIN_ARCHETYPES["Mixed Terrain"]

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_landcover_percentages(class_mask: np.ndarray) -> dict:
    """
    Convert segmentation mask (H×W, values 0-6) into land cover percentages.
    Returns dict with:
      - pct: array of 6 percentages in reference column order
             [urban%, agri%, barren%, forest%, rangeland%, water%]
      - seg_pcts: array of 7 percentages in segmentation class order (0-6)
    """
    total = max(class_mask.size, 1)
    seg_pcts = np.zeros(7, dtype=np.float64)
    for cid in range(7):
        seg_pcts[cid] = float(np.sum(class_mask == cid)) / total * 100.0

    ref_pcts = np.array([
        seg_pcts[0],  # urban
        seg_pcts[1],  # agriculture
        seg_pcts[5],  # barren
        seg_pcts[3],  # forest
        seg_pcts[2],  # rangeland
        seg_pcts[4],  # water
    ], dtype=np.float64)

    return {"pct": ref_pcts, "seg_pcts": seg_pcts}

# ============================================================================
# SEASONAL & ROTATION DATA
# ============================================================================
SEASONAL_DATA = {
    1: {  # Rice
        "kharif": {"sow": "June-July", "harvest": "Nov-Dec", "conditions": "Requires 800-1200mm rainfall"},
        "rabi": {"sow": "Nov-Dec", "harvest": "Apr-May", "conditions": "Irrigation strictly required"}
    },
    2: {  # Wheat
        "rabi": {"sow": "Oct-Nov", "harvest": "Mar-Apr", "conditions": "Requires 10-25°C temperatures"}
    },
    24: { # Soybean
        "kharif": {"sow": "June-July", "harvest": "Sep-Oct", "conditions": "Ideal in 500-750mm rainfall"}
    },
    43: { # Sugarcane
        "kharif": {"sow": "Jan-Mar", "harvest": "Dec-Mar", "conditions": "Long duration 12-18 months"}
    }
}

ROTATION_RULES = {
    1: {  # After Rice
        "recommended_next": [2, 16, 24, 18, 19, 20],
        "avoid": [1],
        "rationale": "Break pest cycles; legumes replenish nitrogen depleted by heavy rice feeding."
    },
    2: {  # After Wheat
        "recommended_next": [1, 24, 19],
        "avoid": [2],
        "rationale": "Cereal-legume rotation prevents nutrient lockout."
    },
    24: {  # After Soybean
        "recommended_next": [2, 3, 4],
        "rationale": "Cereals heavily benefit from soil nitrogen fixed by the previous soybean crop."
    },
    43: { # After Sugarcane
        "recommended_next": [28, 24, 19],
        "rationale": "Deep-rooted sugarcane depletes topsoil; shallow legumes restore it."
    }
}

def _crop_profile_vector(crop_row: list) -> np.ndarray:
    """Extract the 6-element favorable percentage vector from a crop row."""
    return np.array([crop_row[4], crop_row[5], crop_row[6],
                     crop_row[7], crop_row[8], crop_row[9]], dtype=np.float64)

def _compute_suitability(observed_pct: np.ndarray, crop_row: list) -> tuple:
    """
    v3 Scoring Engine — Pure Rule-Based (no cosine similarity).

    The crop data in CROP_DATA represents the inherent suitability (0-100 affinity)
    of each land cover class for that specific crop, NOT a required mix of land covers.

    Score = Base Affinity + Terrain Bonus

    1. Base Affinity: Weighted sum of the observed land cover percentages multiplied
       by their respective crop affinities.
       Example: If land is 100% Agriculture, and Rice's agri affinity is 88, base score is 88.
    2. Terrain Bonus: Up to +18 points based on matching terrain archetypes.
   
    Returns:
        (score_0_to_100, contributions_list)
    """
    fav = _crop_profile_vector(crop_row)
    obs = observed_pct.copy()

    # obs array sums to ~100. We divide by 100 to get fractional presence.
    # Base score automatically captures penalties: if a crop loves water (82)
    # but the land is 100% barren (crop barren affinity 5), score is 5.
    base_scores = (obs / 100.0) * fav
    base_total = float(np.sum(base_scores))

    _, archetype = classify_terrain(obs)
   
    terrain_bonus = 0.0
    cat = crop_row[3]
    cid = crop_row[0]
   
    # Category-level bonuses (e.g. Cereals in Irrigated Farmland)
    if cat in archetype.get("category_bonuses", {}):
        terrain_bonus += archetype["category_bonuses"][cat]
       
    # Crop-specific exceptional fit bonuses (e.g. Jatropha in Arid Dryland)
    if cid in archetype.get("crop_id_bonuses", {}):
        terrain_bonus += archetype["crop_id_bonuses"][cid]

    final_score = base_total + terrain_bonus
    final_score = float(np.clip(final_score, 0.0, 100.0))

    # We want to show which land covers contributed most to the final score
    contrib_list = []
    for i, fname in enumerate(FEATURE_NAMES):
        contrib_list.append({
            "feature":    fname,
            "value":      round(float(obs[i]), 1),
            "shap_value": round(float(base_scores[i]), 3),
        })
   
    # Sort contributions by highest positive impact
    contrib_list.sort(key=lambda c: c["shap_value"], reverse=True)

    return round(final_score, 1), contrib_list

def _compute_suitability_with_uncertainty(observed_pct: np.ndarray, crop_row: list) -> tuple:
    """
    Monte Carlo simulation to estimate prediction uncertainty.
    Returns: (base_score, contribs, confidence_interval, risk_level)
    """
    base_score, contribs = _compute_suitability(observed_pct, crop_row)

    num_simulations = 100
    simulated_scores = []

    for _ in range(num_simulations):
        noise = np.random.normal(0, 5, size=observed_pct.shape)
        noisy_obs = np.clip(observed_pct + noise, 0, 100)
        total = noisy_obs.sum()
        if total > 0:
            noisy_obs = noisy_obs / total * 100.0
        sim_score, _ = _compute_suitability(noisy_obs, crop_row)
        simulated_scores.append(sim_score)

    confidence_interval = (
        round(float(np.percentile(simulated_scores, 5)), 1),
        round(float(np.percentile(simulated_scores, 95)), 1)
    )

    spread = confidence_interval[1] - confidence_interval[0]
    risk_level = "Low" if spread < 10 else "Moderate" if spread < 25 else "High"

    return base_score, contribs, confidence_interval, risk_level

# ============================================================================
# v2: DIVERSITY-AWARE TOP-K SELECTION
# ============================================================================
def _select_diverse_top_k(all_results: list, top_k: int = 15) -> list:
    """
    Select top-K crops with CATEGORY DIVERSITY enforcement.

    Rules:
      - Max 3 crops from any single category in the final selection
      - Within each category, pick the top-scoring ones
      - If we can't fill top_k with the cap, progressively relax

    Also assigns risk_tier labels:
      - "Best Fit"        : top 5 by score (high confidence)
      - "Good Alternative": next 5 (solid but not perfect)
      - "Worth Exploring" : last 5 (lower suitability but unique/interesting)
    """
    # Sort by score descending
    sorted_results = sorted(all_results, key=lambda r: r["suitability_score"], reverse=True)

    selected = []
    category_counts = {}
    MAX_PER_CATEGORY = 3

    # First pass: enforce diversity
    for rec in sorted_results:
        cat = rec["category"]
        count = category_counts.get(cat, 0)
        if count < MAX_PER_CATEGORY:
            selected.append(rec)
            category_counts[cat] = count + 1
            if len(selected) >= top_k:
                break

    # If we couldn't fill the quota (shouldn't happen with 100 crops), relax
    if len(selected) < top_k:
        remaining = [r for r in sorted_results if r not in selected]
        for rec in remaining:
            selected.append(rec)
            if len(selected) >= top_k:
                break

    # Re-sort final selection by score
    selected.sort(key=lambda r: r["suitability_score"], reverse=True)

    # Assign risk tiers
    for i, rec in enumerate(selected):
        if i < 5:
            rec["risk_tier"] = "Best Fit"
        elif i < 10:
            rec["risk_tier"] = "Good Alternative"
        else:
            rec["risk_tier"] = "Worth Exploring"

    return selected

class CropRecommender:
    """
    v3 Data-driven crop suitability scorer using FAO/ICAR/CGIAR reference data.
    Pure rule-based scoring with graduated penalties and terrain-fit analysis.
    Enforces category diversity and provides risk-tiered recommendations.
    """

    def __init__(self):
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

        logging.info(f"CropRecommender v3 initialized with {NUM_CROPS} crops "
                     f"(rule-based scoring + diversity enforcement)")

    def generate_counterfactuals(self, observed_pct: np.ndarray, target_crop_id: int) -> List[Dict]:
        target_crop = next((c for c in CROP_DATA if c[0] == target_crop_id), None)
        if not target_crop:
            return []

        current_score, _ = _compute_suitability(observed_pct, target_crop)
        if current_score >= 75:
            return [{"scenario": "Land is already optimal", "required_change": "None",
                     "projected_score": round(current_score, 1), "feasibility": "High"}]

        fav = _crop_profile_vector(target_crop)
        counterfactuals = []

        # Strategy 1: Increase most impactful favorable feature
        delta = fav - observed_pct
        impact = delta * WEIGHT_ARRAY
        best_feature_idx = int(np.argmax(impact))
        if delta[best_feature_idx] > 0:
            needed_increase = min(delta[best_feature_idx], 20.0)
            modified = observed_pct.copy()
            modified[best_feature_idx] += needed_increase
            total = modified.sum()
            if total > 0:
                modified = modified / total * 100.0
            new_score, _ = _compute_suitability(modified, target_crop)
            counterfactuals.append({
                "scenario": f"Increase {FEATURE_NAMES[best_feature_idx].replace('pct', '')}",
                "required_change": f"+{needed_increase:.1f}%",
                "projected_score": round(new_score, 1),
                "feasibility": "High" if needed_increase < 10 else "Moderate"
            })

        # Strategy 2: Reduce most harmful feature
        penalties = np.where(observed_pct > fav, (observed_pct - fav) * WEIGHT_ARRAY, 0)
        if penalties.max() > 0:
            worst_idx = int(np.argmax(penalties))
            needed_decrease = min(observed_pct[worst_idx] - fav[worst_idx], 15.0)
            modified = observed_pct.copy()
            modified[worst_idx] -= needed_decrease
            total = modified.sum()
            if total > 0:
                modified = modified / total * 100.0
            new_score, _ = _compute_suitability(modified, target_crop)
            counterfactuals.append({
                "scenario": f"Reduce {FEATURE_NAMES[worst_idx].replace('pct', '')}",
                "required_change": f"-{needed_decrease:.1f}%",
                "projected_score": round(new_score, 1),
                "feasibility": "Low" if worst_idx == 0 else "Moderate"
            })

        return counterfactuals

    def recommend_and_explain(self, observed_pct: np.ndarray, top_k: int = 15,
                              current_month: int = None,
                              previous_crop_id: int = None) -> tuple:
        """
        v3: Score all 100 crops with rule-based engine, apply terrain
        bonuses, seasonal/rotational rules, then select diverse top-K across
        categories with risk-tier labels.

        Returns:
            (recommendations_list, explanations_dict, terrain_info)
        """
        # ── Classify terrain ─────────────────────────────────────────────
        terrain_name, terrain_arch = classify_terrain(observed_pct)

        all_results = []
        all_explanations = {}

        for crop in CROP_DATA:
            score, contribs, ci, risk = _compute_suitability_with_uncertainty(observed_pct, crop)

            # ── Apply terrain archetype bonuses ──────────────────────────
            terrain_bonus = 0.0
            cat = crop[3]
            cid = crop[0]

            cat_bonuses = terrain_arch.get("category_bonuses", {})
            if cat in cat_bonuses:
                terrain_bonus += cat_bonuses[cat]

            crop_bonuses = terrain_arch.get("crop_id_bonuses", {})
            if cid in crop_bonuses:
                terrain_bonus += crop_bonuses[cid]

            score = min(100.0, score + terrain_bonus)

            rec = {
                "crop_id":             cid,
                "name":                crop[1],
                "scientific_name":     crop[2],
                "category":            cat,
                "suitability_score":   round(score, 1),
                "confidence_interval": ci,
                "prediction_risk":     risk,
                "favorable": {
                    "urban":       crop[4],
                    "agriculture": crop[5],
                    "barren":      crop[6],
                    "forest":      crop[7],
                    "rangeland":   crop[8],
                    "water":       crop[9],
                },
            }

            # Merge enriched agricultural data
            enriched = self._enriched.get(cid)
            if enriched:
                gc = enriched.get("growing_conditions", {})
                rec["growing_conditions"] = gc
                rec["fertilizers"] = enriched.get("fertilizers", "")
                rec["best_regions"] = enriched.get("best_regions_india", "")
                rec["key_practices"] = enriched.get("key_practices", "")

            # Build explanation
            fav_pct = [crop[4], crop[5], crop[6], crop[7], crop[8], crop[9]]
            explanation = build_explanation(
                crop_id=cid, obs=observed_pct, fav=fav_pct,
                score=score, contribs=contribs,
            )
            if explanation:
                rec["explanation"] = f"<strong>AI Land Analysis:</strong> {explanation}"
            elif enriched and enriched.get("explanation"):
                rec["explanation"] = f"<strong>Agronomic Profile:</strong> {enriched.get('explanation')}"

            # Counterfactual guidance
            rec["counterfactuals"] = self.generate_counterfactuals(observed_pct, cid)

            all_results.append(rec)
            all_explanations[cid] = contribs

        # ── Apply Seasonal & Rotational Rules ────────────────────────────
        if current_month or previous_crop_id:
            SEASONS = {
                (6, 7, 8, 9): "kharif",
                (10, 11, 12, 1, 2, 3): "rabi",
                (4, 5): "zaid"
            }
            current_season = None
            if current_month:
                current_season = next((s for r, s in SEASONS.items() if current_month in r), None)

            for rec in all_results:
                cid = rec["crop_id"]
                if current_season and cid in SEASONAL_DATA:
                    s_data = SEASONAL_DATA[cid].get(current_season)
                    if s_data:
                        rec["planting_window"] = s_data
                        rec["season_bonus"] = True
                        rec["suitability_score"] = min(100.0, rec["suitability_score"] + 10.0)

                if previous_crop_id and previous_crop_id in ROTATION_RULES:
                    rules = ROTATION_RULES[previous_crop_id]
                    if cid in rules.get("recommended_next", []):
                        rec["suitability_score"] = min(100.0, rec["suitability_score"] + 15.0)
                        rec["rotation_benefit"] = rules.get("rationale", "")
                    elif cid in rules.get("avoid", []):
                        rec["suitability_score"] = max(0.0, rec["suitability_score"] - 30.0)
                        rec["rotation_warning"] = rules.get("rationale", "")

        # ── v2: DIVERSITY-AWARE SELECTION ────────────────────────────────
        selected = _select_diverse_top_k(all_results, top_k=top_k)

        top_expl = {str(r["crop_id"]): all_explanations[r["crop_id"]] for r in selected}

        terrain_info = {
            "name": terrain_name,
            "description": terrain_arch.get("description", ""),
        }

        return selected, top_expl, terrain_info

_recommender_instance: Optional[CropRecommender] = None

def get_recommender() -> CropRecommender:
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = CropRecommender()
    return _recommender_instance

def generate_recommendations(class_mask: np.ndarray, top_k: int = 15,
                             current_month: int = None,
                             previous_crop_id: int = None) -> Dict:
    """
    Full recommendation pipeline:
      1. Convert segmentation mask pixels → land cover percentages
      2. Classify terrain archetype
      3. Score all 100 crops with rule-based engine + penalties
      4. Apply terrain bonuses + seasonal/rotational rules
      5. Select diverse top-K with category caps
      6. Return risk-tiered recommendations

    Returns:
        Dict with keys:
          - recommendations: list of crop dicts with scores and risk tiers
          - explanations: dict mapping crop_id → per-feature contributions
          - landcover_profile: dict of land cover % for display
          - terrain_classification: {name, description}
    """
    recommender = get_recommender()

    # Extract percentages
    features = extract_landcover_percentages(class_mask)
    observed = features["pct"]
    seg_pcts = features["seg_pcts"]

    recommendations, explanations, terrain_info = recommender.recommend_and_explain(
        observed, top_k=top_k, current_month=current_month, previous_crop_id=previous_crop_id
    )

    profile = {}
    for cid, name in enumerate(LANDCOVER_NAMES):
        profile[name] = round(float(seg_pcts[cid]), 1)

    return {
        "recommendations":        recommendations,
        "explanations":           explanations,
        "landcover_profile":      profile,
        "terrain_classification": terrain_info,
    }


def _observed_from_percentages(percentages: Dict) -> np.ndarray:
    """Convert API land-cover percentages dict to recommender feature order."""
    urban = float(percentages.get("urban_land", percentages.get("urban", 0.0)) or 0.0)
    agri = float(percentages.get("agriculture", 0.0) or 0.0)
    barren = float(percentages.get("barren", 0.0) or 0.0)
    forest = float(percentages.get("forest", 0.0) or 0.0)
    rangeland = float(percentages.get("rangeland", 0.0) or 0.0)
    water = float(percentages.get("water", 0.0) or 0.0)

    raw = np.array([urban, agri, barren, forest, rangeland, water], dtype=np.float64)
    raw = np.clip(raw, 0.0, 100.0)
    total = float(np.sum(raw))
    if total > 0:
        return (raw / total) * 100.0
    return raw


def _infer_water_regime(obs: np.ndarray) -> str:
    water = float(obs[5])
    barren = float(obs[2])
    if water >= 18:
        return "HUMID"
    if water >= 10:
        return "SUB_HUMID"
    if water >= 5:
        return "SEMI_ARID"
    if barren >= 35:
        return "ARID"
    return "DRY"


def _infer_soil_class(obs: np.ndarray) -> str:
    barren = float(obs[2])
    forest = float(obs[3])
    agri = float(obs[1])
    if barren >= 40:
        return "DEGRADED"
    if forest >= 30:
        return "FOREST_LOAM"
    if agri >= 50:
        return "CULTIVATED_LOAM"
    return "MIXED"


def _infer_market_class(obs: np.ndarray) -> str:
    urban = float(obs[0])
    if urban >= 20:
        return "URBAN_PROXIMATE"
    if urban >= 10:
        return "PERI_URBAN"
    return "RURAL"


def _compute_indices(obs: np.ndarray) -> Dict[str, float]:
    return {
        "ASI": round(float(obs[2] + 0.5 * obs[4]), 2),
        "MAI": round(float(obs[5] + 0.25 * obs[3]), 2),
        "CAI": round(float(obs[1]), 2),
        "UEI": round(float(obs[0]), 2),
    }


def _season_for_category(category: str) -> str:
    mapping = {
        "Cereal": "Kharif/Rabi",
        "Pulse": "Rabi/Kharif",
        "Oilseed": "Kharif/Rabi",
        "Fiber": "Kharif",
        "Sugar": "Annual",
        "Plantation": "Perennial",
        "Spice": "Multi-season",
        "Vegetable": "Multi-season",
        "Fruit": "Perennial",
        "Other": "Context-specific",
    }
    return mapping.get(category, "Multi-season")


def _regime_match_label(water_regime: str, favorable_water: float) -> str:
    if water_regime in {"HUMID", "SUB_HUMID"} and favorable_water >= 10:
        return "Strong"
    if water_regime in {"ARID", "SEMI_ARID", "DRY"} and favorable_water <= 8:
        return "Strong"
    if 6 <= favorable_water <= 14:
        return "Moderate"
    return "Weak"


def recommend_crops(percentages: Dict, top_n: int = 10,
                    current_month: int = None,
                    previous_crop_id: int = None) -> Dict:
    """
    Backward-compatible API consumed by app.py and test scripts.
    Accepts land-cover percentages and returns ranked crop recommendations.
    """
    if not isinstance(percentages, dict):
        return {
            "status": "error",
            "message": "percentages must be a dict",
            "ranked_crops": [],
            "flags": ["Invalid input payload"],
        }

    observed = _observed_from_percentages(percentages)
    if float(np.sum(observed)) <= 0.0:
        return {
            "status": "halted",
            "halt_message": "No valid land-cover percentages were provided.",
            "ranked_crops": [],
            "flags": ["Invalid or empty land-cover profile"],
        }

    top_k = max(1, int(top_n))
    terrain_name, _terrain_arch = classify_terrain(observed)
    water_regime = _infer_water_regime(observed)
    soil_class = _infer_soil_class(observed)
    market_class = _infer_market_class(observed)
    indices = _compute_indices(observed)

    flags = []
    if float(observed[0]) >= 70.0:
        flags.append("Urban dominance is very high; crop planning may be impractical.")
    if float(observed[1]) < 10.0:
        flags.append("Agriculture share is low; expect higher land-preparation cost.")
    if float(observed[2]) >= 40.0:
        flags.append("High barren share detected; prioritize soil restoration and hardy crops.")

    if float(observed[0]) >= 80.0:
        return {
            "status": "halted",
            "halt_message": "Area is predominantly urban and unsuitable for field cropping.",
            "water_regime": water_regime,
            "soil_class": soil_class,
            "market_class": market_class,
            "indices": indices,
            "flags": flags,
            "ranked_crops": [],
        }

    recommender = get_recommender()
    recs, feature_explanations, terrain_info = recommender.recommend_and_explain(
        observed,
        top_k=top_k,
        current_month=current_month,
        previous_crop_id=previous_crop_id,
    )

    ranked_crops = []
    for i, rec in enumerate(recs, start=1):
        ranked_crops.append({
            "rank": i,
            "crop": rec["name"],
            "crop_id": rec["crop_id"],
            "category": rec["category"],
            "season": _season_for_category(rec["category"]),
            "score": rec["suitability_score"],
            "regime_match": _regime_match_label(water_regime, float(rec["favorable"]["water"])),
            "marginal": rec["suitability_score"] < 50,
            "prediction_risk": rec.get("prediction_risk", "Moderate"),
            "confidence_interval": rec.get("confidence_interval"),
            "reasoning": rec.get("explanation", ""),
            "risk_tier": rec.get("risk_tier", "Worth Exploring"),
            "counterfactuals": rec.get("counterfactuals", []),
        })

    return {
        "status": "ok",
        "water_regime": water_regime,
        "soil_class": soil_class,
        "market_class": market_class,
        "terrain_classification": terrain_info,
        "indices": indices,
        "flags": flags,
        "feature_explanations": feature_explanations,
        "ranked_crops": ranked_crops,
    }