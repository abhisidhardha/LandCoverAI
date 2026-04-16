"""
Crop Recommendation Engine v4 — Landscape-Context Scoring + MMR Diversity
100 global crops with land cover suitability percentages sourced from
FAO Agro-Ecological Zones, ICAR crop guidelines, and CGIAR suitability maps.

v4 Changes (April 2026):
  1. Marginal non-agriculture profile — when farmland dominates (60–90%+),
     urban/barren/forest/rangeland/water are renormalized so small class
     differences reshape rankings (not washed out by agri %).
  2. High-ag dampening — agriculture affinity and generic “irrigated block”
     bonuses taper off as ag share rises; non-ag dimensions are amplified.
  3. Terrain archetype order — wetland / forest / arid surface before generic
     irrigated farmland so edge water and tree cover actually steer results.
  4. Top-K: Maximal Marginal Relevance on crop profile embeddings + tighter
     per-category caps so lists are not 15 near-identical cereals.

Prior v3 behavior remains for lower-ag landscapes; v4 mainly fixes repetitive
high-agriculture satellite scenes.
"""

import logging
import hashlib
import math
from datetime import datetime
import numpy as np
from typing import List, Dict, Optional
from crop_explanations import build_explanation, build_evidence_table
from crop_value_profiles import (
    CATEGORY_UI,
    DISPLAY_CATEGORY_ORDER,
    enrich_rec_value_metrics,
    recommendations_by_category,
)

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
# Insertion order is evaluation order: water/forest/arid before generic irrigated tiles.
TERRAIN_ARCHETYPES = {
    "Wetland / Paddy Zone": {
        "condition": lambda obs: obs[5] >= 10 or (obs[1] >= 55 and obs[5] >= 6),
        "description": "Visible surface water — wetland, paddy, and water-adjacent crops gain priority",
        "category_bonuses": {"Fiber": 5, "Vegetable": 3},
        "crop_id_bonuses": {1: 18, 40: 15, 79: 12, 43: 8, 83: 6, 37: 5, 42: 5, 55: 4, 74: 4},
    },
    "Forest / Agroforestry": {
        "condition": lambda obs: obs[3] >= 14,
        "description": "Meaningful tree/forest share — plantation, spice, and shade systems align",
        "category_bonuses": {"Plantation": 15, "Spice": 10, "Fruit": 4},
        "crop_id_bonuses": {45: 12, 46: 15, 47: 14, 48: 14, 50: 15, 52: 12, 53: 15,
                            54: 14, 61: 12, 62: 12, 63: 12, 36: 10, 97: 10, 93: 8,
                            92: 8, 78: 8, 56: 8, 88: 6},
    },
    "Arid Dryland": {
        "condition": lambda obs: obs[2] >= 22 or (obs[4] >= 32 and obs[5] < 12),
        "description": "Semi-arid to arid terrain suited for drought-hardy crops",
        "category_bonuses": {"Cereal": 3, "Pulse": 5, "Oilseed": 4, "Other": 6},
        "crop_id_bonuses": {5: 12, 25: 15, 27: 12, 4: 10, 22: 10, 33: 8, 59: 8,
                            86: 10, 95: 10, 96: 15, 100: 12, 99: 10, 64: 8, 15: 10},
    },
    "Rangeland / Pastoral": {
        "condition": lambda obs: obs[4] >= 26 and obs[1] < 52,
        "description": "Open grassland/shrubland suited for rainfed and hardy crops",
        "category_bonuses": {"Cereal": 5, "Pulse": 6, "Oilseed": 5},
        "crop_id_bonuses": {4: 10, 5: 12, 11: 10, 12: 10, 13: 8, 14: 8, 15: 10,
                            22: 8, 25: 10, 26: 8, 27: 10, 33: 8, 34: 8, 59: 6, 64: 6},
    },
    "Irrigated Farmland": {
        "condition": lambda obs: obs[1] >= 58 and obs[5] < 9 and obs[3] < 15 and obs[2] < 22,
        "description": "Open cultivated plain with limited water, barren, and forest noise",
        "category_bonuses": {"Cereal": 8, "Pulse": 6, "Sugar": 8, "Oilseed": 5, "Fiber": 5, "Vegetable": 4},
    },
    "Mixed Terrain": {
        "condition": lambda obs: True,  # fallback
        "description": "Diverse land cover — multiple crop types viable; marginal context drives fit",
        "category_bonuses": {},
    },
}

def classify_terrain(observed_pct: np.ndarray) -> tuple:
    """
    Classify terrain into an archetype based on observed land cover percentages.
    Returns (archetype_name, archetype_dict).
    Order matters — first match wins (wetland/forest/arid before generic farmland).
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


def _marginal_landscape_match(obs: np.ndarray, fav: np.ndarray) -> float:
    """
    Suitability 0–100 from non-agriculture composition only.
    Both observed and crop profiles are renormalized over {urban, barren, forest, rangeland, water}
    so that when agriculture dominates the scene, remaining classes still differentiate crops.
    """
    non_ag_obs = float(obs[0] + obs[2] + obs[3] + obs[4] + obs[5])
    if non_ag_obs < 1e-6:
        mo = np.ones(5, dtype=np.float64) * 20.0
    else:
        mo = np.array([obs[0], obs[2], obs[3], obs[4], obs[5]], dtype=np.float64)
        mo = mo / non_ag_obs * 100.0

    mc_raw = np.array([fav[0], fav[2], fav[3], fav[4], fav[5]], dtype=np.float64)
    s = float(mc_raw.sum())
    if s < 1e-6:
        mc = np.ones(5, dtype=np.float64) * 20.0
    else:
        mc = mc_raw / s * 100.0

    w = np.array([
        WEIGHTS["urban"], WEIGHTS["barren"], WEIGHTS["forest"],
        WEIGHTS["rangeland"], WEIGHTS["water"],
    ], dtype=np.float64)
    w = w / (np.mean(w) + 1e-9)
    d = np.abs(mo - mc) * w
    return float(100.0 - np.sqrt(np.mean(d ** 2)))


def _crop_embedding_norm(crop_row: list) -> np.ndarray:
    """Unit vector over the 6 land-cover affinities for diversity (MMR)."""
    v = _crop_profile_vector(crop_row).astype(np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def _compute_suitability(observed_pct: np.ndarray, crop_row: list) -> tuple:
    """
    v4 FIXED Scoring Engine — Removes agri bias, enforces hard constraints,
    adds positive bonuses, and uses similarity instead of dot-product.
    """

    obs = observed_pct.astype(float)
    fav = _crop_profile_vector(crop_row).astype(float)
    crop_id = crop_row[0]

    # Normalize just in case
    if obs.sum() > 0:
        obs = obs / obs.sum() * 100.0

    # ------------------------------------------------------------------
    # 1. HARD CONSTRAINT FILTER (CRITICAL FIX)
    # ------------------------------------------------------------------
    # Water dependent crops: keep rice as soft-gated (penalize later),
    # keep hard filter for other highly water-dependent crops.
    if fav[5] >= 40 and obs[5] < 8:
        if crop_id != 1:
            return 0.0, []

    # Forest crops (coffee, tea, spices) use a soft penalty later instead
    # of a hard filter so they can still appear as low-score options.
    if fav[3] >= 50 and obs[3] < 10:
        pass

    # Arid crops should not be in high water areas
    if fav[2] >= 35 and obs[5] > 20:
        return 0.0, []

    # ------------------------------------------------------------------
    # 2. SIMILARITY + MARGINAL LANDSCAPE (high-ag farmland de-clustering)
    # ------------------------------------------------------------------
    ag_pct = float(obs[1])
    marginal_blend = float(np.clip((ag_pct - 48.0) / 34.0, 0.0, 1.0))
    high_ag_damp = 1.0 - 0.62 * marginal_blend

    diff = np.abs(obs - fav)
    per_dim = np.ones(6, dtype=np.float64)
    amp = 1.0 + 1.28 * marginal_blend
    per_dim[0] *= amp
    per_dim[2] *= amp
    per_dim[3] *= amp
    per_dim[4] *= amp
    per_dim[5] *= amp
    per_dim[1] *= max(0.28, 1.12 - 0.95 * marginal_blend)

    weighted_diff = diff * WEIGHT_ARRAY * per_dim
    similarity_score = 100.0 - np.sqrt(np.mean(weighted_diff ** 2))

    marginal_match = _marginal_landscape_match(obs, fav)
    agri_factor = (obs[1] * fav[1]) ** 0.45
    agri_weight = 0.20 * max(0.04, 1.0 - 1.18 * marginal_blend)
    similarity_score = (1.0 - agri_weight) * similarity_score + agri_weight * agri_factor
    mix_m = 0.52 * marginal_blend
    similarity_score = (1.0 - mix_m) * similarity_score + mix_m * marginal_match

    # ------------------------------------------------------------------
    # 3. POSITIVE BONUSES
    # ------------------------------------------------------------------
    bonus = 0.0

    # Peri-urban / mosaic signal (urban share matters when farm dominates)
    if marginal_blend > 0.22 and obs[0] >= 6.0 and fav[0] >= 3.0:
        bonus += min(7.0, 0.45 * obs[0] + 0.25 * fav[0]) * marginal_blend

    # Strong water match -> boost rice-type crops (reduced)
    if obs[5] >= 12 and fav[5] >= 40:
        bonus += 12

    # Moderate water crops
    elif obs[5] >= 8 and fav[5] >= 20:
        bonus += 6

    # Dryland match
    if obs[2] >= 30 and fav[2] >= 35:
        bonus += 8

    # Forest match (lower bar when ag dominates — small canopy % still steers)
    forest_obs_min = 20.0 if marginal_blend < 0.45 else 10.0
    if obs[3] >= forest_obs_min and fav[3] >= 38:
        bonus += 8

    range_obs_min = 25.0 if marginal_blend < 0.45 else 14.0
    if obs[4] >= range_obs_min and fav[4] >= 30:
        bonus += 5

    # Generic irrigated-plain bonus (taper off when every scene looks “high ag”)
    if obs[1] >= 70 and obs[5] < 5:
        if fav[1] >= 80 and fav[5] <= 5:
            bonus += 10 * high_ag_damp
        elif fav[1] >= 75 and fav[5] <= 5:
            bonus += 6 * high_ag_damp
        elif fav[1] >= 70 and fav[5] <= 3:
            bonus += 4 * high_ag_damp

    # ------------------------------------------------------------------
    # 5. SIGNATURE CROP BOOSTS (IMPORTANT FOR RULE SYSTEM)
    # ------------------------------------------------------------------
    # Jute
    if crop_id == 40 and obs[5] >= 12:
        bonus += 10

    # Taro (water crop)
    if crop_id == 79 and obs[5] >= 15:
        bonus += 10

    # Bajra / arid crops
    if crop_id == 5 and obs[2] >= 35:
        bonus += 10

    # Signature row-crops (damp on repetitive high-ag mosaics)
    if crop_id == 2 and obs[1] >= 70 and obs[5] < 8:
        bonus += 12 * high_ag_damp

    if crop_id == 3 and obs[1] >= 65 and obs[5] < 10:
        bonus += 10 * high_ag_damp

    if crop_id == 29 and obs[1] >= 70 and obs[5] < 8:
        bonus += 9 * high_ag_damp

    if crop_id == 38 and obs[1] >= 70 and obs[5] < 5:
        bonus += 9 * high_ag_damp

    if crop_id == 43 and obs[1] >= 75 and obs[5] < 10:
        bonus += 8 * high_ag_damp

    if crop_id == 24 and obs[1] >= 70 and obs[5] < 8:
        bonus += 8 * high_ag_damp

    # Rice - context-aware bonus tiers
    if crop_id == 1:
        if obs[5] >= 15:
            bonus += 15
        elif obs[5] >= 8:
            bonus += 10
        elif obs[1] >= 85 and obs[5] >= 2:
            bonus += 12
        elif obs[1] >= 80 and obs[5] >= 1.5:
            bonus += 8

    # ------------------------------------------------------------------
    # 6. PENALTIES (LIGHT — NOT DOMINANT)
    # ------------------------------------------------------------------
    penalty = 0.0

    # If water exists but crop doesn't like water
    if obs[5] > 10 and fav[5] < 5:
        penalty += (obs[5] - fav[5]) * 0.8

    # Too barren for sensitive crops
    if obs[2] > 30 and fav[2] < 10:
        penalty += (obs[2] - fav[2]) * 0.5

    # Heavy penalty for highly water-loving crops in low-water terrain
    # (instead of hard filtering to zero).
    if fav[5] >= 40 and obs[5] < 8:
        water_deficit = fav[5] - obs[5]
        if crop_id == 1:
            # Rice can still be viable with irrigation in high-agriculture areas,
            # so apply a much softer low-water penalty.
            penalty += (water_deficit ** 1.1) * 0.05
            if obs[1] < 75:
                penalty += 8
        else:
            penalty += (water_deficit ** 1.5) * 0.3
            if obs[1] < 75:
                penalty += 15

    # Agriculture over-matching penalty only when crop requirements do not
    # fit dry/open farmland conditions.
    if obs[1] >= 70:
        if fav[5] >= 15 and obs[5] < 5:
            penalty += (fav[5] - obs[5]) * 0.5
        if fav[3] >= 30 and obs[3] < 5:
            penalty += (fav[3] - obs[3]) * 0.4

    # Heavy penalty for forest crops in non-forest terrain (soft filter).
    if fav[3] >= 50 and obs[3] < 10:
        forest_gap = fav[3] - obs[3]
        penalty += forest_gap * 0.8

    # Special rice penalty when visible water is below recommended level.
    if crop_id == 1 and obs[5] < 8:
        water_gap = 8 - obs[5]
        penalty += water_gap ** 1.3

    # ------------------------------------------------------------------
    # 7. FINAL SCORE
    # ------------------------------------------------------------------
    raw_score = similarity_score + bonus - penalty

    # Apply non-linear spreading to reduce clustering.
    if raw_score > 50:
        final_score = 50 + ((raw_score - 50) ** 0.85)
    elif raw_score > 0:
        final_score = raw_score ** 1.15
    else:
        final_score = 0.0

    # Deterministic micro-variation to avoid identical ranks.
    category = crop_row[3]
    category_offsets = {
        "Cereal": 0.5,
        "Oilseed": 0.4,
        "Pulse": 0.35,
        "Fiber": 0.3,
        "Sugar": 0.25,
        "Vegetable": 0.2,
        "Plantation": 0.15,
        "Spice": 0.1,
        "Fruit": 0.05,
        "Other": 0.0,
    }
    id_offset = (crop_id * 0.1) % 0.5
    final_score = final_score + category_offsets.get(category, 0.0) + id_offset

    final_score = float(np.clip(final_score, 0, 100))

    # ------------------------------------------------------------------
    # 8. CONTRIBUTIONS (for explainability)
    # ------------------------------------------------------------------
    contrib_list = []
    for i, fname in enumerate(FEATURE_NAMES):
        contrib_list.append({
            "feature": fname,
            "value": round(float(obs[i]), 1),
            "shap_value": round(float((100 - diff[i]) * WEIGHT_ARRAY[i]), 3),
        })

    contrib_list.sort(key=lambda c: c["shap_value"], reverse=True)

    return round(final_score, 1), contrib_list

def _compute_suitability_with_uncertainty(observed_pct: np.ndarray, crop_row: list,
                                          climate_features: Optional[Dict] = None) -> tuple:
    """
    Monte Carlo simulation to estimate prediction uncertainty.
    Returns: (base_score, contribs, confidence_interval, risk_level)
    """
    base_score, contribs = _compute_suitability(observed_pct, crop_row)
    climate_score = climate_suitability_score(int(crop_row[0]), climate_features)
    climate_component = _climate_normalized_score(climate_score)
    adjusted_score = float(np.clip((0.70 * base_score) + (0.30 * climate_component), 0.0, 100.0))

    num_simulations = 100
    simulated_scores = []

    for _ in range(num_simulations):
        noise = np.random.normal(0, 2, size=observed_pct.shape)
        noisy_obs = np.clip(observed_pct + noise, 0, 100)
        total = noisy_obs.sum()
        if total > 0:
            noisy_obs = noisy_obs / total * 100.0
        sim_score, _ = _compute_suitability(noisy_obs, crop_row)
        simulated_scores.append(float(np.clip((0.70 * sim_score) + (0.30 * climate_component), 0.0, 100.0)))

    confidence_interval = (
        round(float(np.percentile(simulated_scores, 5)), 1),
        round(float(np.percentile(simulated_scores, 95)), 1)
    )

    spread = confidence_interval[1] - confidence_interval[0]
    risk_level = "Low" if spread < 10 else "Moderate" if spread < 25 else "High"

    return round(adjusted_score, 1), contribs, confidence_interval, risk_level

# ============================================================================
# v4: MMR + category caps (land-cover embedding diversity)
# ============================================================================
def _select_diverse_top_k(all_results: list, top_k: int = 15, observed_pct: np.ndarray = None) -> list:
    """
    Maximal Marginal Relevance on unit-normalized crop land-cover profiles,
    with a per-category cap so top-K is not dominated by near-duplicate cereals.

    Relevance: suitability_score. Redundancy: max cosine similarity to picks.
    """
    sorted_results = sorted(all_results, key=lambda r: r["suitability_score"], reverse=True)

    viable_crops = [r for r in sorted_results if r["suitability_score"] >= 20.0]
    if len(viable_crops) < top_k:
        viable_crops = sorted_results[:max(top_k, len(sorted_results))]

    pool_size = min(len(viable_crops), max(top_k * 5, 55))
    pool = viable_crops[:pool_size]

    emb_by_id = {row[0]: _crop_embedding_norm(row) for row in CROP_DATA}
    lambda_mmr = 0.74
    max_per_cat = 3

    selected: List[Dict] = []
    selected_ids = set()
    category_counts: Dict[str, int] = {}
    sel_embs: List[np.ndarray] = []

    def _max_cos_sim(emb: np.ndarray) -> float:
        if not sel_embs:
            return 0.0
        return max(float(np.dot(emb, s)) for s in sel_embs)

    while len(selected) < top_k:
        best_rec = None
        best_mmr = -1e9
        for rec in pool:
            cid = rec["crop_id"]
            if cid in selected_ids:
                continue
            emb = emb_by_id.get(cid)
            if emb is None:
                continue
            rel = float(rec["suitability_score"]) / 100.0
            mmr = lambda_mmr * rel - (1.0 - lambda_mmr) * _max_cos_sim(emb)
            cat = rec["category"]
            over = category_counts.get(cat, 0) - max_per_cat + 1
            if over > 0:
                mmr -= 0.24 * over
            if mmr > best_mmr:
                best_mmr = mmr
                best_rec = rec
        if best_rec is None:
            break
        selected.append(best_rec)
        selected_ids.add(best_rec["crop_id"])
        cat = best_rec["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        sel_embs.append(emb_by_id[best_rec["crop_id"]])

    if len(selected) < top_k:
        for rec in sorted_results:
            if rec["crop_id"] in selected_ids:
                continue
            selected.append(rec)
            selected_ids.add(rec["crop_id"])
            if len(selected) >= top_k:
                break

    selected.sort(key=lambda r: r["suitability_score"], reverse=True)

    for rec in selected:
        score = rec["suitability_score"]
        if score >= 75:
            rec["risk_tier"] = "Best Fit"
        elif score >= 55:
            rec["risk_tier"] = "Good Alternative"
        else:
            rec["risk_tier"] = "Worth Exploring"

    return selected


STRUCTURED_FLAGS = {
    "urban_dominance": {
        "condition": lambda obs: obs[0] >= 70,
        "severity": "critical",
        "message": "Urban land >70% - crop viability is very limited.",
        "remediation": "Consider rooftop/container farming or periurban plots.",
    },
    "low_agriculture": {
        "condition": lambda obs: obs[1] < 10,
        "severity": "high",
        "message": "Farmland share <10% - soil preparation cost will be high.",
        "remediation": "Prioritize land clearing and organic matter addition before sowing.",
    },
    "high_barren": {
        "condition": lambda obs: obs[2] >= 40,
        "severity": "high",
        "message": "Barren land >40% - salinity/hardpan risk is elevated.",
        "remediation": "Test EC and pH. Consider gypsum application for saline soils.",
    },
    "no_water_source": {
        "condition": lambda obs: obs[5] < 2 and obs[2] >= 20,
        "severity": "moderate",
        "message": "No visible water bodies in an arid landscape.",
        "remediation": "Confirm groundwater depth; drip irrigation is recommended.",
    },
}


def _build_structured_flags(observed_pct: np.ndarray) -> List[Dict]:
    flags = []
    for name, spec in STRUCTURED_FLAGS.items():
        if spec["condition"](observed_pct):
            flags.append({
                "name": name,
                "severity": spec["severity"],
                "message": spec["message"],
                "remediation": spec["remediation"],
            })
    return flags


def _build_input_flag(name: str, severity: str, message: str, remediation: str) -> List[Dict]:
    return [{
        "name": name,
        "severity": severity,
        "message": message,
        "remediation": remediation,
    }]

class CropRecommender:
    """
    v4 Crop suitability: FAO/ICAR/CGIAR reference profiles + marginal non-ag
    landscape matching under high farmland share + MMR-diverse shortlists.
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

        logging.info(f"CropRecommender v4 initialized with {NUM_CROPS} crops "
                     f"(marginal landscape scoring + MMR top-K)")

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
                              previous_crop_id: int = None,
                              climate_features: Optional[Dict] = None) -> tuple:
        """
        v3: Score all 100 crops with rule-based engine, apply terrain
        bonuses, seasonal/rotational rules, then select diverse top-K across
        categories with risk-tier labels.

        Returns:
            (recommendations_list, explanations_dict, terrain_info, recommendations_by_cat)
        """
        # ── Classify terrain ─────────────────────────────────────────────
        terrain_name, terrain_arch = classify_terrain(observed_pct)

        all_results = []
        all_explanations = {}

        for crop in CROP_DATA:
            climate_score = climate_suitability_score(crop[0], climate_features)
            score, contribs, ci, risk = _compute_suitability_with_uncertainty(
                observed_pct, crop, climate_features=climate_features
            )
            fav = _crop_profile_vector(crop)

            # ── Apply terrain archetype bonuses ──────────────────────────
            terrain_bonus = 0.0
            cat = crop[3]
            cid = crop[0]

            cat_bonuses = terrain_arch.get("category_bonuses", {})
            if cat in cat_bonuses:
                if terrain_name == "Irrigated Farmland":
                    if fav[1] >= 70:
                        terrain_bonus += cat_bonuses[cat] * 0.5
                    else:
                        terrain_bonus += cat_bonuses[cat] * 0.2
                else:
                    terrain_bonus += cat_bonuses[cat] * 0.5

            crop_bonuses = terrain_arch.get("crop_id_bonuses", {})
            if cid in crop_bonuses:
                terrain_bonus += crop_bonuses[cid] * 0.4

            score = min(100.0, score + terrain_bonus)
            ci_adjusted = (
                round(min(100.0, float(ci[0]) + terrain_bonus), 1),
                round(min(100.0, float(ci[1]) + terrain_bonus), 1),
            )
            evidence_table = build_evidence_table(observed_pct, [crop[4], crop[5], crop[6], crop[7], crop[8], crop[9]], contribs, score)

            rec = {
                "crop_id":             cid,
                "name":                crop[1],
                "scientific_name":     crop[2],
                "category":            cat,
                "suitability_score":   round(score, 1),
                "confidence_interval": ci_adjusted,
                "prediction_risk":     risk,
                "terrain_bonus_pts":   round(float(terrain_bonus), 1),
                "climate_bonus_pts":   round(float(climate_score), 1),
                "climate_features":    climate_features,
                "favorable": {
                    "urban":       crop[4],
                    "agriculture": crop[5],
                    "barren":      crop[6],
                    "forest":      crop[7],
                    "rangeland":   crop[8],
                    "water":       crop[9],
                },
                "evidence_table": evidence_table,
                "terrain_name": terrain_name,
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
                ci=ci, risk=risk,
                terrain_name=terrain_name,
                terrain_bonus=terrain_bonus,
                crop_name=crop[1],
                climate_features=climate_features,
                climate_score=climate_score,
            )
            if explanation:
                rec["explanation"] = f"<strong>AI Land Analysis:</strong> {explanation}"
            elif enriched and enriched.get("explanation"):
                rec["explanation"] = f"<strong>Agronomic Profile:</strong> {enriched.get('explanation')}"

            # Counterfactual guidance
            rec["counterfactuals"] = self.generate_counterfactuals(observed_pct, cid)
            rec["explanation_meta"] = {
                "engine_version": "v4.0",
                "scoring_method": "marginal_landscape_and_weighted_affinity_mmr",
                "terrain_detected": terrain_name,
                "terrain_bonus_pts": round(float(terrain_bonus), 1),
                "climate_bonus_pts": round(float(climate_score), 1),
                "climate_features": climate_features,
                "ci_low": ci_adjusted[0],
                "ci_high": ci_adjusted[1],
                "uncertainty_level": risk,
                "timestamp_utc": datetime.utcnow().isoformat(),
                "obs_hash": hashlib.md5(np.asarray(observed_pct, dtype=np.float64).tobytes()).hexdigest()[:8],
            }

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
                        ci = rec.get("confidence_interval")
                        if ci and len(ci) == 2:
                            rec["confidence_interval"] = (
                                round(min(100.0, float(ci[0]) + 5.0), 1),
                                round(min(100.0, float(ci[1]) + 5.0), 1),
                            )
                        rec["suitability_score"] = min(100.0, rec["suitability_score"] + 5.0)

                if previous_crop_id and previous_crop_id in ROTATION_RULES:
                    rules = ROTATION_RULES[previous_crop_id]
                    if cid in rules.get("recommended_next", []):
                        ci = rec.get("confidence_interval")
                        if ci and len(ci) == 2:
                            rec["confidence_interval"] = (
                                round(min(100.0, float(ci[0]) + 8.0), 1),
                                round(min(100.0, float(ci[1]) + 8.0), 1),
                            )
                        rec["suitability_score"] = min(100.0, rec["suitability_score"] + 8.0)
                        rec["rotation_benefit"] = rules.get("rationale", "")
                    elif cid in rules.get("avoid", []):
                        ci = rec.get("confidence_interval")
                        if ci and len(ci) == 2:
                            rec["confidence_interval"] = (
                                round(max(0.0, float(ci[0]) - 30.0), 1),
                                round(max(0.0, float(ci[1]) - 30.0), 1),
                            )
                        rec["suitability_score"] = max(0.0, rec["suitability_score"] - 30.0)
                        rec["rotation_warning"] = rules.get("rationale", "")

        # ── Yield / market indices + practical score (after final land score) ──
        for rec in all_results:
            enrich_rec_value_metrics(rec)

        recommendations_by_cat = recommendations_by_category(all_results, per_category=2)

        # ── v2: DIVERSITY-AWARE SELECTION ────────────────────────────────
        selected = _select_diverse_top_k(all_results, top_k=top_k, observed_pct=observed_pct)

        # Filter out very low-score crops from display output.
        MIN_DISPLAY_SCORE = 25.0
        selected = [r for r in selected if r["suitability_score"] >= MIN_DISPLAY_SCORE]

        if len(selected) < 5:
            selected = sorted(all_results, key=lambda r: r["suitability_score"], reverse=True)[:5]

        # DEBUG: score distribution for tuning verification.
        all_scores = sorted([r["suitability_score"] for r in all_results], reverse=True)
        logging.info(f"Score distribution - Top 20: {all_scores[:20]}")
        logging.info(f"Score distribution - Bottom 10: {all_scores[-10:]}")
        logging.info(f"Crops above 50%%: {len([s for s in all_scores if s >= 50])}")
        logging.info(f"Crops above 70%%: {len([s for s in all_scores if s >= 70])}")

        top_expl = {str(r["crop_id"]): all_explanations[r["crop_id"]] for r in selected}

        terrain_info = {
            "name": terrain_name,
            "description": terrain_arch.get("description", ""),
        }

        return selected, top_expl, terrain_info, recommendations_by_cat

_recommender_instance: Optional[CropRecommender] = None

def get_recommender() -> CropRecommender:
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = CropRecommender()
    return _recommender_instance

def generate_recommendations(class_mask: np.ndarray, top_k: int = 15,
                             current_month: int = None,
                             previous_crop_id: int = None,
                             climate_features: Optional[Dict] = None) -> Dict:
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

    recommendations, explanations, terrain_info, by_cat = recommender.recommend_and_explain(
        observed, top_k=top_k, current_month=current_month, previous_crop_id=previous_crop_id,
        climate_features=climate_features
    )

    profile = {}
    for cid, name in enumerate(LANDCOVER_NAMES):
        profile[name] = round(float(seg_pcts[cid]), 1)

    water_regime = _infer_water_regime(observed)
    category_sections = _build_category_sections_payload(by_cat, water_regime)

    return {
        "recommendations":         recommendations,
        "explanations":            explanations,
        "landcover_profile":       profile,
        "terrain_classification":  terrain_info,
        "climate_features":        climate_features,
        "category_sections":       category_sections,
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


def climate_suitability_score(crop_id: int, features: Optional[Dict]) -> float:
    """
    Score a crop using climate, elevation, and soil context.

    Returns a value roughly in the range -30 to +30.
    """
    if not isinstance(features, dict) or not features:
        return 0.0

    score = 0.0
    rainfall = features.get("rainfall_mm")
    temp_avg = features.get("temp_avg")
    temp_min = features.get("temp_min")
    temp_max = features.get("temp_max")
    elevation = features.get("elevation_m")
    soil_type = str(features.get("soil_type") or "").lower()
    agro_zone = str(features.get("agro_zone") or "").lower()

    dryland_ids = {4, 5, 11, 12, 13, 14, 15, 16, 22, 25, 26, 27, 29, 31, 33, 34, 35, 59, 64, 95, 96, 100}
    humid_ids = {1, 36, 37, 40, 43, 45, 46, 47, 48, 50, 52, 53, 54, 55, 56, 61, 62, 63, 82, 83, 84, 92, 93, 97, 98}
    cool_ids = {2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 23, 44, 66, 67, 68, 70, 71, 72, 80, 81, 88, 94}
    warm_ids = {1, 3, 4, 5, 19, 20, 21, 24, 28, 30, 31, 32, 38, 39, 43, 49, 55, 56, 57, 58, 60, 66, 68, 69, 73, 74, 75, 77, 82, 83, 84, 85, 89, 90, 91, 92, 93, 94, 96, 99, 100}

    if rainfall is not None:
        rainfall = float(rainfall)
        if crop_id == 1:
            score += 25 if rainfall >= 1000 else 15 if rainfall >= 800 else 5 if rainfall >= 600 else -20
        elif crop_id == 2:
            score += 20 if 500 <= rainfall <= 750 else 10 if 400 <= rainfall <= 900 else -15
        elif crop_id == 5:
            score += 20 if rainfall < 600 else 10 if rainfall < 800 else -10
        elif crop_id == 28:
            score += 18 if 500 <= rainfall <= 750 else 8 if rainfall < 1000 else -10
        elif crop_id == 40:
            score += 25 if rainfall >= 1500 else 10 if rainfall >= 1200 else -25
        elif crop_id == 43:
            score += 25 if rainfall >= 1200 else 15 if rainfall >= 1000 else -20
        elif crop_id in {45, 46, 47, 48, 50, 52, 53, 54, 61, 62, 63}:
            score += 15 if rainfall >= 1400 else 8 if rainfall >= 1000 else -10
        elif crop_id in {96, 99, 100}:
            score += 15 if rainfall <= 600 else 8 if rainfall <= 900 else -12
        elif crop_id in dryland_ids:
            score += 10 if rainfall <= 800 else -8

    if temp_avg is not None:
        temp_avg = float(temp_avg)
        if crop_id == 1:
            score += 10 if 22 <= temp_avg <= 32 else -5 if temp_avg < 18 else 0
        elif crop_id == 2:
            score += 15 if 15 <= temp_avg <= 25 else -20 if temp_avg > 30 else 0
        elif crop_id in {45, 46}:
            score += 20 if 15 <= temp_avg <= 25 else -15
        elif crop_id == 83:
            score += 12 if 20 <= temp_avg <= 32 else -8
        elif crop_id == 88:
            score += 15 if 5 <= temp_avg <= 20 else -12
        elif crop_id in {53, 54, 52, 61, 62, 63}:
            score += 15 if 15 <= temp_avg <= 28 else -10
        elif crop_id in warm_ids:
            score += 8 if 20 <= temp_avg <= 34 else -8 if temp_avg < 15 else 0
        elif crop_id in cool_ids:
            score += 8 if 10 <= temp_avg <= 25 else -10 if temp_avg > 30 else 0

    if temp_min is not None and crop_id in {2, 45, 46, 88}:
        temp_min = float(temp_min)
        if crop_id == 2 and temp_min < 0:
            score -= 8
        elif crop_id in {45, 46} and temp_min < 10:
            score -= 5
        elif crop_id == 88 and temp_min < -5:
            score -= 10

    if temp_max is not None and crop_id in {1, 2, 45, 46, 88}:
        temp_max = float(temp_max)
        if crop_id == 2 and temp_max > 35:
            score -= 8
        elif crop_id in {45, 46} and temp_max > 32:
            score -= 5
        elif crop_id == 88 and temp_max > 35:
            score -= 8

    if elevation is not None:
        elevation = float(elevation)
        if crop_id == 45:
            score += 20 if 600 <= elevation <= 2000 else 10 if elevation >= 300 else -15
        elif crop_id == 46:
            score += 20 if 600 <= elevation <= 1800 else -15
        elif crop_id == 88:
            score += 20 if elevation >= 1000 else -10
        elif crop_id in {53, 54}:
            score += 15 if 600 <= elevation <= 1500 else -10
        elif crop_id in {52, 61, 62, 63}:
            score += 10 if elevation >= 300 else -5
        elif crop_id in {1, 2, 28, 43}:
            score += 5 if elevation < 1000 else -5
        elif crop_id in {96, 99, 100}:
            score += 10 if elevation < 700 else 0

    if soil_type:
        if soil_type in {"black", "clay"}:
            if crop_id in {1, 38, 43, 44, 66, 70, 71}:
                score += 10
        elif soil_type in {"red", "loamy"}:
            if crop_id in {2, 5, 16, 17, 18, 19, 20, 21, 24, 28, 29, 30, 31, 32, 57, 58, 59, 60, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 82, 83, 84, 85, 89, 90}:
                score += 8
        elif soil_type == "sandy":
            if crop_id in {5, 15, 25, 27, 31, 33, 34, 35, 59, 64, 96, 99, 100}:
                score += 10
        elif soil_type == "laterite":
            if crop_id in {45, 46, 47, 48, 50, 52, 53, 54, 61, 62, 63, 97}:
                score += 10

    if agro_zone:
        if agro_zone in {"tropical_wet", "humid_subtropical"}:
            if crop_id in humid_ids:
                score += 6
        elif agro_zone == "sub_humid":
            if crop_id in humid_ids or crop_id in {3, 28, 29, 30, 38, 43, 57, 82, 83, 84, 89, 90}:
                score += 4
        elif agro_zone in {"semi_arid", "arid"}:
            if crop_id in dryland_ids:
                score += 8
            if crop_id in {1, 40, 43, 45, 46, 47, 48, 50, 52, 53, 54, 61, 62, 63, 83}:
                score -= 8

    if crop_id in {1, 43, 40, 83, 50, 52, 53, 54, 45, 46, 47, 48} and agro_zone in {"tropical_wet", "humid_subtropical"}:
        score += 4
    if crop_id in {5, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 57, 58, 59, 60} and agro_zone in {"semi_arid", "arid"}:
        score += 5

    return float(max(-30.0, min(30.0, score)))


def _climate_normalized_score(climate_score: float) -> float:
    return float(np.clip(50.0 + climate_score, 0.0, 100.0))


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


def _risk_tier_from_land_score(score: float) -> str:
    if score >= 75:
        return "Best Fit"
    if score >= 55:
        return "Good Alternative"
    return "Worth Exploring"


def _ranked_crop_item_from_rec(rec: dict, rank: int, water_regime: str) -> dict:
    """API-shaped dict aligned with recommend_crops ranked_crops entries."""
    fav = rec.get("favorable") or {}
    land_w = float(fav.get("water", 0))
    score = float(rec.get("suitability_score") or 0)
    return {
        "rank": rank,
        "crop": rec["name"],
        "crop_id": rec["crop_id"],
        "scientific_name": rec.get("scientific_name", ""),
        "category": rec["category"],
        "season": _season_for_category(rec["category"]),
        "score": rec["suitability_score"],
        "yield_potential": rec.get("yield_potential"),
        "market_demand": rec.get("market_demand"),
        "practical_score": rec.get("practical_score"),
        "recommendation_labels": rec.get("recommendation_labels", []),
        "regime_match": _regime_match_label(water_regime, land_w),
        "marginal": score < 50,
        "prediction_risk": rec.get("prediction_risk", "Moderate"),
        "confidence_interval": rec.get("confidence_interval"),
        "reasoning": rec.get("explanation", ""),
        "risk_tier": rec.get("risk_tier") or _risk_tier_from_land_score(score),
        "counterfactuals": rec.get("counterfactuals", []),
        "growing_conditions": rec.get("growing_conditions", {}),
        "fertilizers": rec.get("fertilizers", ""),
        "best_regions": rec.get("best_regions", ""),
        "key_practices": rec.get("key_practices", ""),
        "favorable": fav,
        "evidence_table": rec.get("evidence_table", []),
        "explanation_meta": rec.get("explanation_meta", {}),
        "terrain_bonus_pts": rec.get("terrain_bonus_pts", 0.0),
        "climate_bonus_pts": rec.get("climate_bonus_pts", 0.0),
        "terrain_name": rec.get("terrain_name", ""),
    }


def _build_category_sections_payload(by_cat: Dict[str, List[dict]], water_regime: str) -> list:
    sections: List[dict] = []
    for cat in DISPLAY_CATEGORY_ORDER:
        picks = by_cat.get(cat) or []
        if not picks:
            continue
        title, subtitle = CATEGORY_UI.get(cat, (cat, ""))
        sections.append({
            "category": cat,
            "section_title": title,
            "section_subtitle": subtitle,
            "picks": [
                _ranked_crop_item_from_rec(r, j + 1, water_regime)
                for j, r in enumerate(picks)
            ],
        })
    return sections


def recommend_crops(percentages: Dict, top_n: int = 10,
                    current_month: int = None,
                    previous_crop_id: int = None,
                    climate_features: Optional[Dict] = None) -> Dict:
    """
    Backward-compatible API consumed by app.py and test scripts.
    Accepts land-cover percentages and returns ranked crop recommendations.
    """
    if not isinstance(percentages, dict):
        return {
            "status": "error",
            "message": "percentages must be a dict",
            "ranked_crops": [],
            "flags": _build_input_flag(
                "invalid_input", "high", "percentages must be a dict", "Provide a dictionary of land-cover percentages."
            ),
        }

    observed = _observed_from_percentages(percentages)
    if float(np.sum(observed)) <= 0.0:
        return {
            "status": "halted",
            "halt_message": "No valid land-cover percentages were provided.",
            "ranked_crops": [],
            "flags": _build_input_flag(
                "empty_profile", "high", "No valid land-cover percentages were provided.",
                "Provide non-zero values for at least one land-cover class."
            ),
        }

    top_k = max(1, int(top_n))
    terrain_name, _terrain_arch = classify_terrain(observed)
    water_regime = _infer_water_regime(observed)
    soil_class = _infer_soil_class(observed)
    market_class = _infer_market_class(observed)
    indices = _compute_indices(observed)

    flags = []
    if float(observed[0]) >= 70.0:
        flags.append({
            "name": "urban_dominance",
            "severity": "critical",
            "message": "Urban dominance is very high; crop planning may be impractical.",
            "remediation": "Consider rooftop/container farming or periurban plots.",
        })
    if float(observed[1]) < 10.0:
        flags.append({
            "name": "low_agriculture",
            "severity": "high",
            "message": "Agriculture share is low; expect higher land-preparation cost.",
            "remediation": "Prioritize land clearing and organic matter addition before sowing.",
        })
    if float(observed[2]) >= 40.0:
        flags.append({
            "name": "high_barren",
            "severity": "high",
            "message": "High barren share detected; prioritize soil restoration and hardy crops.",
            "remediation": "Test EC and pH. Consider gypsum application for saline soils.",
        })

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
    recs, feature_explanations, terrain_info, by_cat = recommender.recommend_and_explain(
        observed,
        top_k=top_k,
        current_month=current_month,
        previous_crop_id=previous_crop_id,
        climate_features=climate_features,
    )

    ranked_crops = [
        _ranked_crop_item_from_rec(rec, i, water_regime)
        for i, rec in enumerate(recs, start=1)
    ]
    category_sections = _build_category_sections_payload(by_cat, water_regime)

    return {
        "status": "ok",
        "water_regime": water_regime,
        "soil_class": soil_class,
        "market_class": market_class,
        "terrain_classification": terrain_info,
        "indices": indices,
        "flags": flags,
        "feature_explanations": feature_explanations,
        "climate_features": climate_features,
        "ranked_crops": ranked_crops,
        "category_sections": category_sections,
    }