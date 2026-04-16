"""
Relative yield potential and market depth indices (0–100) for recommendation UX.
Heuristic India-centric tiers: staples score high on market depth; spices/plantation
on value; arid/niche crops moderated. Not a futures price model — guides sorting
and on-screen labels beside land-cover suitability.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

Override = Union[Tuple[int, int], Tuple[int, int, str]]

CATEGORY_DEFAULTS: Dict[str, Tuple[int, int]] = {
    "Cereal": (88, 94),
    "Pulse": (76, 86),
    "Oilseed": (80, 84),
    "Fiber": (78, 80),
    "Sugar": (90, 88),
    "Plantation": (72, 76),
    "Spice": (62, 90),
    "Vegetable": (82, 87),
    "Fruit": (74, 82),
    "Other": (70, 72),
}

DISPLAY_CATEGORY_ORDER: List[str] = [
    "Cereal", "Pulse", "Oilseed", "Fiber", "Sugar",
    "Plantation", "Spice", "Vegetable", "Fruit", "Other",
]

CATEGORY_UI: Dict[str, Tuple[str, str]] = {
    "Cereal": ("Cereals & millets", "Staple grains — deep markets when land and water fit"),
    "Pulse": ("Pulses & legumes", "Rotation-friendly legumes with steady wholesale demand"),
    "Oilseed": ("Oilseeds", "Edible oils — market depth linked to processing capacity"),
    "Fiber": ("Fiber crops", "Industrial offtake — water and soil matter alongside land"),
    "Sugar": ("Sugar crops", "High biomass — strong yield where irrigation/rainfall align"),
    "Plantation": ("Plantation & perennials", "Long-cycle — tree cover and niche buyers drive fit"),
    "Spice": ("Spices & condiments", "High value per hectare — climate-sensitive"),
    "Vegetable": ("Vegetables & tubers", "Short cycles — urban proximity lifts market scores"),
    "Fruit": ("Fruits", "Orchards — establishment vs. sustained mandi/export demand"),
    "Other": ("Other crops", "Special-purpose — verify buyers and agronomy locally"),
}

# Crop-specific tweaks; IDs match CROP_DATA. Optional 3rd str: specialty | export | regional
CROP_VALUE_OVERRIDES: Dict[int, Override] = {
    1: (92, 96),
    2: (90, 95), 3: (92, 92), 4: (84, 82), 5: (78, 78), 6: (76, 80),
    7: (82, 85), 8: (78, 82), 9: (74, 78), 10: (76, 80), 11: (68, 60, "regional"),
    12: (75, 76), 13: (72, 74), 14: (73, 75), 15: (74, 76),
    16: (78, 85), 17: (76, 80), 18: (77, 84), 19: (79, 83), 20: (78, 82),
    21: (80, 85), 22: (77, 79), 23: (79, 82), 24: (86, 88),
    25: (70, 72), 26: (72, 74), 27: (74, 78),
    28: (87, 86), 29: (85, 88), 30: (84, 86), 31: (78, 82), 32: (76, 80),
    33: (80, 76), 34: (76, 78), 35: (72, 72, "regional"), 36: (82, 88),
    37: (78, 86),
    38: (83, 86), 39: (78, 78), 40: (85, 74), 41: (72, 75), 42: (74, 72),
    43: (93, 90), 44: (88, 82),
    45: (70, 78), 46: (68, 80), 47: (72, 79), 48: (74, 77), 49: (78, 83),
    50: (76, 84), 51: (73, 82),
    52: (58, 88, "specialty"), 53: (55, 86, "specialty"), 54: (56, 87, "specialty"),
    55: (80, 85), 56: (81, 84), 57: (84, 88), 58: (79, 84), 59: (76, 82),
    60: (78, 80), 61: (54, 84, "specialty"), 62: (52, 83, "specialty"),
    63: (48, 86, "specialty"), 64: (38, 97, "specialty"),
    65: (77, 80),
    66: (88, 90), 67: (87, 89), 68: (86, 90), 69: (84, 88), 70: (83, 87),
    71: (83, 87), 72: (82, 88), 73: (84, 86), 74: (80, 84), 75: (82, 86),
    76: (80, 80), 77: (82, 82), 78: (76, 78), 79: (79, 82),
    80: (82, 86), 81: (83, 86),
    82: (78, 88), 83: (86, 86), 84: (82, 84), 85: (80, 82), 86: (76, 84),
    87: (75, 86), 88: (72, 82), 89: (80, 87), 90: (82, 86),
    91: (74, 78), 92: (81, 84), 93: (70, 84), 94: (68, 88, "specialty"),
    95: (72, 82), 96: (78, 80), 97: (76, 78), 98: (78, 82),
    99: (77, 78), 100: (74, 68, "regional"),
}


def _resolve_profile(crop_id: int, category: str) -> Tuple[int, int, Optional[str]]:
    o = CROP_VALUE_OVERRIDES.get(crop_id)
    if o is not None:
        if len(o) == 3:
            return int(o[0]), int(o[1]), str(o[2])
        return int(o[0]), int(o[1]), None
    y, m = CATEGORY_DEFAULTS.get(category, (72, 74))
    return int(y), int(m), None


def _build_labels(land: float, y: int, m: int, niche: Optional[str]) -> List[Dict[str, str]]:
    labels: List[Dict[str, str]] = []
    if land >= 72:
        labels.append({"id": "land_fit", "text": "Strong land match"})
    elif land >= 55:
        labels.append({"id": "land_fit", "text": "Good land match"})
    if y >= 86:
        labels.append({"id": "yield", "text": "High yield potential"})
    elif y >= 76:
        labels.append({"id": "yield", "text": "Solid yield potential"})
    if m >= 92:
        labels.append({"id": "market", "text": "Deep market / high liquidity"})
    elif m >= 80:
        labels.append({"id": "market", "text": "Strong market demand"})
    if niche == "export":
        labels.append({"id": "niche", "text": "Export-oriented crop"})
    elif niche == "specialty":
        labels.append({"id": "niche", "text": "Specialty / premium segment"})
    elif niche == "regional":
        labels.append({"id": "niche", "text": "Regional / niche market"})
    return labels


def enrich_rec_value_metrics(rec: dict) -> None:
    """Mutates rec with yield_potential, market_demand, practical_score, recommendation_labels."""
    land = float(rec.get("suitability_score") or 0.0)
    cid = int(rec["crop_id"])
    cat = str(rec.get("category") or "Other")
    y, m, niche = _resolve_profile(cid, cat)
    rec["yield_potential"] = y
    rec["market_demand"] = m
    rec["practical_score"] = round(0.52 * land + 0.26 * y + 0.22 * m, 1)
    rec["recommendation_labels"] = _build_labels(land, y, m, niche)
    meta = rec.get("explanation_meta")
    if isinstance(meta, dict):
        meta["value_profile_version"] = "1.0"


def recommendations_by_category(
    all_results: List[dict],
    per_category: int = 2,
    min_land_score: float = 26.0,
) -> Dict[str, List[dict]]:
    """
    Top per_category crops per display category by practical_score (land + yield + market).
    Only categories with at least one viable crop are included.
    """
    buckets: Dict[str, List[dict]] = {c: [] for c in DISPLAY_CATEGORY_ORDER}
    for rec in all_results:
        if float(rec.get("suitability_score") or 0) < min_land_score:
            continue
        cat = str(rec.get("category") or "Other")
        if cat not in buckets:
            cat = "Other"
        buckets[cat].append(rec)

    out: Dict[str, List[dict]] = {}
    for cat in DISPLAY_CATEGORY_ORDER:
        lst = buckets.get(cat, [])
        if not lst:
            continue
        lst.sort(key=lambda r: float(r.get("practical_score") or 0), reverse=True)
        out[cat] = lst[: max(1, int(per_category))]
    return out
