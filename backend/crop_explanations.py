"""
Crop-Specific Explanation Builders
===================================
Each builder receives:
    obs   : np.ndarray of 6 floats  — observed land cover %
            index order: [urban, agri, barren, forest, rangeland, water]
    fav   : list of 6 ints          — crop's favorable % (same order)
    score : float                   — final suitability score 0-100
    contribs : list of dicts        — per-feature SHAP-style contributions

Returns:
    str — plain-text explanation (no HTML tags)

Design principles
-----------------
* Every sentence references actual observed numbers, not generic agronomic text.
* Each builder knows the crop's specific growth requirements and maps observed
  land cover to those requirements explicitly.
* Risk flags are raised only when an observed value actually warrants it.
* The explanation reads as a reasoning chain: land → agronomic factor → verdict.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict

# Column index constants (matching FEATURE_NAMES / ref column order)
I_URBAN     = 0
I_AGRI      = 1
I_BARREN    = 2
I_FOREST    = 3
I_RANGE     = 4
I_WATER     = 5


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _pct(val: float) -> str:
    """Round and format a percentage value for display."""
    return f"{round(val, 1)}%"


def _confidence_phrase(score: float) -> str:
    """Convert numeric score to a farmer-facing confidence phrase."""
    if score >= 80:
        return "very well suited"
    elif score >= 65:
        return "well suited"
    elif score >= 50:
        return "moderately suited"
    elif score >= 35:
        return "marginally suited"
    else:
        return "poorly suited"


def _dominant_driver(contribs: List[Dict]) -> Dict:
    """Return the single contribution with the highest absolute positive value."""
    positive = [c for c in contribs if c["shap_value"] > 0]
    if not positive:
        return contribs[0]
    return max(positive, key=lambda c: c["shap_value"])


def _worst_risk(contribs: List[Dict]) -> Dict | None:
    """Return the most negative contribution, or None if no negatives exist."""
    negative = [c for c in contribs if c["shap_value"] < 0]
    if not negative:
        return None
    return min(negative, key=lambda c: c["shap_value"])


# ---------------------------------------------------------------------------
# Crop 1 — Rice (Paddy)
# ---------------------------------------------------------------------------

def explain_rice(obs: np.ndarray, fav: list, score: float,
                 contribs: List[Dict]) -> str:
    agri  = obs[I_AGRI]
    water = obs[I_WATER]
    barren = obs[I_BARREN]
    urban  = obs[I_URBAN]

    parts = []

    if agri >= 60:
        parts.append(
            f"With {_pct(agri)} of the surrounding area as established farmland, "
            f"the soil is already worked and suitable for paddy cultivation — "
            f"this is the strongest signal in your land profile."
        )
    elif agri >= 35:
        parts.append(
            f"About {_pct(agri)} of your area is active farmland, providing a "
            f"reasonable agricultural base for rice cultivation."
        )
    else:
        parts.append(
            f"Farmland covers only {_pct(agri)} of your area, which is lower than "
            f"rice prefers (ideally above 50%). Soil preparation will be important."
        )

    if water >= 15:
        parts.append(
            f"Natural water coverage at {_pct(water)} strongly favours rice — "
            f"paddy needs standing water for 80% of its growth cycle, and natural "
            f"moisture nearby significantly reduces your irrigation investment."
        )
    elif water >= 5:
        parts.append(
            f"Water bodies cover {_pct(water)} of the area, indicating natural "
            f"moisture availability. Rice will need irrigation supplementation, "
            f"but drainage infrastructure costs are moderate."
        )
    else:
        parts.append(
            f"Water coverage is low at {_pct(water)}. Rice is viable but will "
            f"depend entirely on canal or groundwater irrigation, which increases "
            f"input cost. Verify irrigation access before committing."
        )

    if barren >= 25:
        parts.append(
            f"A significant {_pct(barren)} of the area is barren or degraded land — "
            f"this is a risk flag for rice. Barren patches often indicate poor drainage, "
            f"salt accumulation, or hardpan soils. Get a soil salinity test near "
            f"these patches before planting, especially at field edges."
        )
    elif barren >= 12:
        parts.append(
            f"Barren land at {_pct(barren)} is a moderate concern. "
            f"A soil check near dry patches is advisable before the Kharif season."
        )

    if urban >= 20:
        parts.append(
            f"Urban area at {_pct(urban)} can restrict natural drainage during "
            f"heavy monsoon — waterlogging beyond the paddy field's need is a risk. "
            f"Check for low-lying collection points near the farm."
        )

    verdict = _confidence_phrase(score)
    parts.append(
        f"Overall, your land is {verdict} for rice (score {round(score)}/100). "
        f"Best season: Kharif (June–November)."
    )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 2 — Wheat
# ---------------------------------------------------------------------------

def explain_wheat(obs: np.ndarray, fav: list, score: float,
                  contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    barren = obs[I_BARREN]
    water  = obs[I_WATER]
    forest = obs[I_FOREST]
    range_ = obs[I_RANGE]

    parts = []

    if agri >= 65:
        parts.append(
            f"Your area has a strong agricultural base at {_pct(agri)}, "
            f"which is the primary requirement for wheat — it needs well-managed "
            f"soil with assured irrigation or residual moisture."
        )
    elif agri >= 40:
        parts.append(
            f"Farmland covers {_pct(agri)} of the area — a moderate base for wheat. "
            f"This is sufficient for Rabi wheat with proper soil preparation."
        )
    else:
        parts.append(
            f"Active farmland is limited at {_pct(agri)}, which is the main "
            f"constraint for wheat here. Wheat is a high-input, high-management crop "
            f"that performs best on established cultivated land."
        )

    if range_ >= 20:
        parts.append(
            f"Rangeland at {_pct(range_)} supports wheat suitability — "
            f"semi-arid grassland areas with residual soil moisture are viable "
            f"for dryland wheat with supplemental irrigation."
        )

    if barren >= 20:
        parts.append(
            f"Barren land at {_pct(barren)} is a clear risk for wheat. "
            f"Wheat is significantly more salt-sensitive than rice or millets — "
            f"if these dry patches carry saline soil, wheat yield will drop sharply. "
            f"A soil EC test is essential before sowing."
        )
    elif barren >= 10:
        parts.append(
            f"Barren coverage of {_pct(barren)} is a moderate salinity concern. "
            f"Consider soil testing if barren patches are near the intended field."
        )

    if water >= 10:
        parts.append(
            f"Water coverage at {_pct(water)} is higher than wheat typically needs — "
            f"wheat does not tolerate waterlogged conditions. "
            f"Ensure adequate drainage, particularly for Rabi season sowing."
        )

    if forest >= 20:
        parts.append(
            f"Forest cover at {_pct(forest)} is not compatible with wheat's need "
            f"for full sun and open fields. Wheat cannot tolerate shade "
            f"and forest soils are typically acidic, which reduces yield."
        )

    verdict = _confidence_phrase(score)
    parts.append(
        f"Overall, your land is {verdict} for wheat (score {round(score)}/100). "
        f"Best season: Rabi (October–April)."
    )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 5 — Pearl Millet / Bajra
# ---------------------------------------------------------------------------

def explain_pearl_millet(obs: np.ndarray, fav: list, score: float,
                         contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water  = obs[I_WATER]
    forest = obs[I_FOREST]
    urban  = obs[I_URBAN]

    parts = []

    if barren >= 30:
        parts.append(
            f"Barren land at {_pct(barren)} is actually a positive signal for "
            f"pearl millet — unlike most cereals, bajra is specifically designed "
            f"for dry, sandy, and degraded soils where other crops fail. "
            f"This is where bajra earns its place."
        )
    elif barren >= 15:
        parts.append(
            f"Barren land coverage of {_pct(barren)} suits pearl millet well. "
            f"Bajra's deep root system can extract moisture from 1.5m depth in "
            f"dry sandy soils — it thrives where rainfall is below 400mm."
        )

    if range_ >= 35:
        parts.append(
            f"Rangeland at {_pct(range_)} aligns well with bajra's growing conditions. "
            f"Open semi-arid grassland with light sandy soils is bajra's home territory."
        )
    elif range_ >= 20:
        parts.append(
            f"Rangeland at {_pct(range_)} provides additional viable area for bajra "
            f"cultivation under rainfed conditions."
        )

    if agri >= 40:
        parts.append(
            f"The {_pct(agri)} agricultural base further strengthens the case — "
            f"existing farmland means better access, soil history, and lower preparation cost."
        )
    elif agri >= 15:
        parts.append(
            f"Farmland at {_pct(agri)} provides a managed cultivation base "
            f"alongside the dryland areas suitable for bajra."
        )

    if water >= 8:
        parts.append(
            f"Water body coverage at {_pct(water)} is notable — pearl millet "
            f"does not benefit from wetland proximity and cannot survive waterlogging. "
            f"Avoid low-lying waterlogged fields for bajra."
        )

    if forest >= 15:
        parts.append(
            f"Forest cover at {_pct(forest)} is a mild constraint — bajra needs "
            f"full sun and does not perform well in partially shaded or humid conditions."
        )

    if urban >= 30:
        parts.append(
            f"High urban coverage ({_pct(urban)}) limits the land available for "
            f"dryland bajra cultivation, though peri-urban fields can still be viable."
        )

    verdict = _confidence_phrase(score)
    parts.append(
        f"Overall, your land is {verdict} for pearl millet (score {round(score)}/100). "
        f"Best season: Kharif (July–October). "
        f"Key advantage: if other crops are failing due to dry conditions here, "
        f"bajra is the most likely to succeed."
    )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 43 — Sugarcane
# ---------------------------------------------------------------------------

def explain_sugarcane(obs: np.ndarray, fav: list, score: float,
                      contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    water  = obs[I_WATER]
    barren = obs[I_BARREN]
    urban  = obs[I_URBAN]
    range_ = obs[I_RANGE]

    parts = []

    if agri >= 70:
        parts.append(
            f"Your area has an exceptional agricultural base at {_pct(agri)} — "
            f"sugarcane demands prime managed farmland above all else. "
            f"It is a 12–18 month crop requiring continuous soil fertility, "
            f"and a strong existing agricultural base is the most critical condition."
        )
    elif agri >= 50:
        parts.append(
            f"Farmland at {_pct(agri)} provides a solid base for sugarcane. "
            f"This is above the viable threshold, though sugarcane performs best "
            f"when agricultural land exceeds 70% of the surrounding area."
        )
    elif agri >= 30:
        parts.append(
            f"Agricultural coverage at {_pct(agri)} is below sugarcane's ideal range. "
            f"Sugarcane is a very high-input crop — limited farmland means limited "
            f"suitable field area and likely reduced access to irrigation infrastructure."
        )
    else:
        parts.append(
            f"Agricultural land at only {_pct(agri)} is a significant limitation — "
            f"sugarcane is not recommended here unless the specific field itself is "
            f"prime irrigated farmland."
        )

    if water >= 5:
        parts.append(
            f"Water presence at {_pct(water)} supports irrigation availability — "
            f"sugarcane requires 1,500–2,500mm of water equivalent per season, "
            f"making proximity to water bodies a meaningful operational advantage."
        )
    else:
        parts.append(
            f"Water body coverage is low at {_pct(water)}. Sugarcane's high water "
            f"demand will need to be met entirely through canal or borewell irrigation."
        )

    if barren >= 15:
        parts.append(
            f"Barren land at {_pct(barren)} is a concern for sugarcane. "
            f"Sugarcane is a heavy nutrient feeder and does not tolerate degraded soils."
        )

    if urban >= 20:
        parts.append(
            f"Urban area at {_pct(urban)} fragments the landscape — sugarcane mills "
            f"require minimum farm sizes to be economically viable due to haulage "
            f"constraints. Scattered fields in a peri-urban setting may not "
            f"meet mill sourcing requirements."
        )

    if range_ >= 25:
        parts.append(
            f"Rangeland at {_pct(range_)} does not support sugarcane cultivation — "
            f"this crop cannot be grown on unmanaged grassland soils."
        )

    verdict = _confidence_phrase(score)
    parts.append(
        f"Overall, your land is {verdict} for sugarcane (score {round(score)}/100). "
        f"Crop duration: 12–18 months. Ratoon crops possible for up to 3 cycles."
    )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 16 — Chickpea / Chana
# ---------------------------------------------------------------------------

def explain_chickpea(obs: np.ndarray, fav: list, score: float,
                     contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water  = obs[I_WATER]
    forest = obs[I_FOREST]

    parts = []

    if agri >= 55:
        parts.append(
            f"Active farmland at {_pct(agri)} is the primary driver here — "
            f"chickpea grows best on existing cultivated soil with residual moisture "
            f"from the Kharif season."
        )
    elif agri >= 35:
        parts.append(
            f"Farmland at {_pct(agri)} provides an adequate base for chickpea. "
            f"Residual moisture in post-Kharif soil is chickpea's primary water source."
        )
    else:
        parts.append(
            f"Agricultural land is limited at {_pct(agri)}. Chickpea can still grow "
            f"on dryland areas, but management complexity increases."
        )

    if barren >= 20 or range_ >= 25:
        combined = barren + range_
        parts.append(
            f"Barren land ({_pct(barren)}) and rangeland ({_pct(range_)}) together "
            f"cover {_pct(combined)} of the area — this is manageable for chickpea, "
            f"which tolerates semi-arid dryland conditions better than most pulses."
        )

    parts.append(
        f"A key agronomic advantage: chickpea fixes atmospheric nitrogen via Rhizobium "
        f"bacteria. This reduces fertiliser cost for the following season's crop."
    )

    if water >= 8:
        parts.append(
            f"Water body coverage at {_pct(water)} is a caution signal. "
            f"Chickpea is extremely sensitive to waterlogging — even 24 hours of "
            f"standing water at the root zone can cause plant death. "
            f"Avoid low-lying fields near water bodies for this crop."
        )
    elif water >= 3:
        parts.append(
            f"Water coverage at {_pct(water)} is marginal — choose elevated or "
            f"well-drained field positions to avoid waterlogging risk."
        )

    if forest >= 15:
        parts.append(
            f"Forest cover at {_pct(forest)} is a negative indicator — chickpea "
            f"needs full sunlight and open ventilation."
        )

    verdict = _confidence_phrase(score)
    parts.append(
        f"Overall, your land is {verdict} for chickpea (score {round(score)}/100)."
    )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 3 — Maize / Corn
# ---------------------------------------------------------------------------

def explain_maize(obs: np.ndarray, fav: list, score: float,
                  contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    range_ = obs[I_RANGE]
    forest = obs[I_FOREST]
    water  = obs[I_WATER]
    barren = obs[I_BARREN]
    urban  = obs[I_URBAN]

    parts = []

    if agri >= 60:
        parts.append(
            f"With {_pct(agri)} established farmland surrounding your area, "
            f"maize has an excellent cultivation base — the soil is managed, "
            f"accessible, and responsive to high nitrogen inputs."
        )
    elif agri >= 35:
        parts.append(
            f"Active farmland at {_pct(agri)} provides a workable base for maize."
        )
    else:
        parts.append(
            f"Agricultural land is limited at {_pct(agri)}, which constrains maize "
            f"performance."
        )

    if range_ >= 15:
        parts.append(
            f"Rangeland at {_pct(range_)} is a useful signal for maize — "
            f"semi-open grassland soils with good drainage can support maize."
        )

    if 5 <= forest <= 20:
        parts.append(f"Forest coverage at {_pct(forest)} is within maize's compatible range.")
    elif forest > 20:
        parts.append(
            f"Forest at {_pct(forest)} starts to become a constraint due to shading "
            f"and humidity."
        )

    if water >= 10:
        parts.append(
            f"Water body coverage at {_pct(water)} is a drainage alert for maize. "
            f"Maize seedlings die within 48 hours of waterlogging."
        )

    if barren >= 20:
        parts.append(f"Barren land at {_pct(barren)} indicates nutrient-poor soils.")

    if urban >= 15:
        parts.append(
            f"Urban coverage at {_pct(urban)} is not a barrier for maize — "
            f"peri-urban maize for poultry feed or wet market supply is strong."
        )

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for maize (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 4 — Sorghum / Jowar
# ---------------------------------------------------------------------------

def explain_sorghum(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water  = obs[I_WATER]
    forest = obs[I_FOREST]

    parts = []

    if barren >= 25 and range_ >= 25:
        parts.append(
            f"Your area has {_pct(barren)} barren land and {_pct(range_)} rangeland — "
            f"together these are a strong positive signal for sorghum."
        )
    elif barren >= 25:
        parts.append(
            f"Barren land at {_pct(barren)} is a positive signal for sorghum — "
            f"jowar specifically targets the dry soils that dominate such zones."
        )
    elif range_ >= 30:
        parts.append(f"Rangeland at {_pct(range_)} strongly favours sorghum.")
    elif barren >= 12 or range_ >= 15:
        parts.append(f"Dry land coverage is compatible with sorghum's drought-hardy profile.")

    if agri >= 50:
        parts.append(f"Established farmland at {_pct(agri)} further strengthens the case.")
    elif agri >= 25:
        parts.append(f"Agricultural land at {_pct(agri)} provides a managed base.")

    if barren >= 20 or agri < 35:
        parts.append(
            f"A key advantage: sorghum can ratoon — the stubble after first "
            f"harvest re-grows a second crop without replanting."
        )

    if water >= 10:
        parts.append(f"Water body coverage at {_pct(water)} is a concern — field drainage must be guaranteed.")

    if forest >= 15:
        parts.append(f"Forest cover at {_pct(forest)} is a constraint — sorghum needs full sun.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for sorghum (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 24 — Soybean
# ---------------------------------------------------------------------------

def explain_soybean(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    forest = obs[I_FOREST]
    barren = obs[I_BARREN]
    water  = obs[I_WATER]
    range_ = obs[I_RANGE]

    parts = []

    if agri >= 65:
        parts.append(
            f"Agricultural land at {_pct(agri)} is an excellent foundation for soybean — "
            f"it demands well-managed, structured soil."
        )
    elif agri >= 45:
        parts.append(f"Farmland at {_pct(agri)} provides a solid base for soybean.")
    elif agri >= 25:
        parts.append(f"Agricultural coverage at {_pct(agri)} is on the lower side for soybean.")
    else:
        parts.append(f"Farmland at only {_pct(agri)} is a significant constraint for soybean.")

    if 8 <= forest <= 25:
        parts.append(f"Forest coverage at {_pct(forest)} is actually compatible with soybean's early growth.")
    elif forest > 25:
        parts.append(f"Forest at {_pct(forest)} exceeds what soybean can tolerate.")

    if barren >= 15:
        parts.append(f"Barren land at {_pct(barren)} is a meaningful risk for soybean due to poor Rhizobium nodulation.")

    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} is marginally compatible.")

    if water >= 8:
        parts.append(
            f"Water bodies covering {_pct(water)} of the area raise drainage concerns. "
            f"Soybean roots cannot survive more than 48 hours of waterlogging."
        )

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for soybean (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 28 — Groundnut / Peanut
# ---------------------------------------------------------------------------

def explain_groundnut(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water  = obs[I_WATER]
    forest = obs[I_FOREST]

    parts = []

    if agri >= 60:
        parts.append(f"Agricultural land at {_pct(agri)} is a strong base for groundnut.")
    elif agri >= 35:
        parts.append(f"Farmland at {_pct(agri)} gives groundnut a reasonable cultivation base.")
    else:
        parts.append(f"Agricultural land at {_pct(agri)} is limited.")

    if barren >= 15:
        parts.append(f"Barren land at {_pct(barren)} is a conditional positive for groundnut if sandy.")

    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} is compatible with groundnut on light dryland soils.")

    if water >= 8:
        parts.append(
            f"Water body coverage at {_pct(water)} is a serious concern for groundnut. "
            f"Periodic waterlogging causes aflatoxin contamination and pod rot."
        )
    elif water >= 3:
        parts.append(f"Water coverage at {_pct(water)} warrants drainage verification.")

    if forest >= 15:
        parts.append(f"Forest land at {_pct(forest)} is a negative signal — soils tend to be heavy.")

    parts.append(f"Critical management note: gypsum application at the pegging stage is non-negotiable.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for groundnut (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 29 — Mustard / Rapeseed
# ---------------------------------------------------------------------------

def explain_mustard(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    range_ = obs[I_RANGE]
    barren = obs[I_BARREN]
    water  = obs[I_WATER]
    forest = obs[I_FOREST]

    parts = []

    if agri >= 65:
        parts.append(f"Strong agricultural coverage at {_pct(agri)} is the primary driver for mustard.")
    elif agri >= 40:
        parts.append(f"Agricultural land at {_pct(agri)} provides a viable base for mustard.")
    else:
        parts.append(f"Farmland at {_pct(agri)} is on the lower side but mustard remains viable.")

    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} is a meaningful positive for mustard.")

    if barren >= 15:
        parts.append(f"Barren land at {_pct(barren)} is manageable for mustard.")

    if water >= 8:
        parts.append(
            f"Water body coverage at {_pct(water)} is a drainage concern for mustard. "
            f"Mustard does not tolerate waterlogging."
        )

    if forest >= 12:
        parts.append(f"Forest cover at {_pct(forest)} is a negative for mustard — it needs full winter sun.")

    parts.append(
        f"Sulphur fertilisation (40 kg/ha as sulphate of ammonia or gypsum) is the "
        f"single most yield- and quality-determining input for mustard."
    )

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for mustard (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 38 — Cotton (American Upland / Bt)
# ---------------------------------------------------------------------------

def explain_cotton(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    barren = obs[I_BARREN]
    water  = obs[I_WATER]
    forest = obs[I_FOREST]
    range_ = obs[I_RANGE]
    urban  = obs[I_URBAN]

    parts = []

    if agri >= 65:
        parts.append(
            f"Agricultural land at {_pct(agri)} is a strong signal for cotton — "
            f"Bt cotton requires structured, well-drained, managed soil."
        )
    elif agri >= 40:
        parts.append(f"Farmland at {_pct(agri)} gives cotton a viable base.")
    elif agri >= 20:
        parts.append(f"Agricultural coverage at {_pct(agri)} is below cotton's comfort zone.")
    else:
        parts.append(f"Farmland at only {_pct(agri)} makes cotton a difficult recommendation.")

    if 5 <= barren <= 20:
        parts.append(
            f"Barren land at {_pct(barren)} is within cotton's tolerance range — "
            f"cotton's tap root reaches 1.5–2m depth, allowing it to access sub-soil moisture."
        )
    elif barren > 20:
        parts.append(f"Barren land at {_pct(barren)} is exceeding cotton's tolerance.")

    if water >= 10:
        parts.append(
            f"Water coverage at {_pct(water)} is a boll rot alert for cotton. "
            f"Raised bed planting and field drainage are mandatory."
        )

    if range_ >= 15:
        parts.append(f"Rangeland at {_pct(range_)} is marginally compatible.")

    if forest >= 10:
        parts.append(f"Forest coverage at {_pct(forest)} is a constraint — cotton needs warm, sunny conditions.")

    if urban >= 10:
        parts.append(
            f"Urban presence at {_pct(urban)} is a market access advantage for proximity "
            f"to ginning mills."
        )

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for cotton (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 40 — Jute
# ---------------------------------------------------------------------------

def explain_jute(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    water  = obs[I_WATER]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    forest = obs[I_FOREST]

    parts = []

    if water >= 12:
        parts.append(
            f"Water body coverage at {_pct(water)} is the most critical positive signal "
            f"for jute, needed not just for growing, but for retting."
        )
    elif water >= 5:
        parts.append(
            f"Water coverage at {_pct(water)} is moderate — retting infrastructure "
            f"must be confirmed before committing to jute cultivation."
        )
    else:
        parts.append(
            f"Water coverage at only {_pct(water)} is the primary constraint for jute. "
            f"Without accessible slow-moving water, the fibre cannot be economically extracted."
        )

    if agri >= 60:
        parts.append(f"Agricultural land at {_pct(agri)} supports jute well.")
    elif agri >= 35:
        parts.append(f"Farmland at {_pct(agri)} provides a workable base for jute.")
    else:
        parts.append(f"Agricultural coverage at {_pct(agri)} is limited for jute.")

    if barren >= 10:
        parts.append(f"Barren land at {_pct(barren)} is a meaningful negative for jute.")

    if range_ >= 15:
        parts.append(f"Rangeland at {_pct(range_)} is not suitable for jute.")

    if 5 <= forest <= 15:
        parts.append(f"Forest cover at {_pct(forest)} is within tolerance.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for jute (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 66 — Potato
# ---------------------------------------------------------------------------

def explain_potato(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    urban  = obs[I_URBAN]
    water  = obs[I_WATER]
    barren = obs[I_BARREN]
    forest = obs[I_FOREST]

    parts = []

    if agri >= 60:
        parts.append(f"Agricultural coverage at {_pct(agri)} is an excellent foundation for potato.")
    elif agri >= 35:
        parts.append(f"Farmland at {_pct(agri)} provides a workable base for potato.")
    else:
        parts.append(f"Agricultural land at {_pct(agri)} is limited for potato.")

    if urban >= 10:
        parts.append(
            f"Urban coverage at {_pct(urban)} is a commercial advantage for potato — "
            f"proximity to urban markets and cold storage reduces post-harvest loss."
        )

    if water >= 8:
        parts.append(
            f"Water body coverage at {_pct(water)} is a drainage alert for potato. "
            f"Potato tubers in waterlogged soil develop tuber rot within 48 hours."
        )
    elif 2 <= water < 8:
        parts.append(
            f"Water coverage at {_pct(water)} is within a workable range — potato needs "
            f"consistent irrigation but must never sit in waterlogged conditions."
        )

    if barren >= 15:
        parts.append(f"Barren land at {_pct(barren)} is a moderate concern due to soil compaction risk.")

    if 5 <= forest <= 20:
        parts.append(
            f"Forest coverage at {_pct(forest)} suggests a cooler microclimate, which "
            f"is ideal for potato."
        )
    elif forest > 20:
        parts.append(f"Forest at {_pct(forest)} is high, potentially increasing late blight risk if it suggests humidity.")

    parts.append(
        f"Late blight is the most economically damaging disease for potato — a preventive "
        f"fungicide programme is essential."
    )

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for potato (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 67 — Onion
# ---------------------------------------------------------------------------

def explain_onion(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    water  = obs[I_WATER]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    urban  = obs[I_URBAN]
    forest = obs[I_FOREST]

    parts = []

    if agri >= 60:
        parts.append(f"Agricultural land at {_pct(agri)} is a strong foundation for onion.")
    elif agri >= 35:
        parts.append(f"Farmland at {_pct(agri)} provides a workable base for onion.")
    else:
        parts.append(f"Agricultural land at {_pct(agri)} is limited for onion.")

    if water >= 8:
        parts.append(
            f"Water body coverage at {_pct(water)} is a bulb quality alert for onion. "
            f"Waterlogging during bulb development causes neck rot and purple blotch."
        )
    elif water >= 3:
        parts.append(
            f"Water presence at {_pct(water)} is moderate — withhold irrigation completely "
            f"in the final 15 days before harvest to allow proper neck drying."
        )

    if barren >= 15:
        parts.append(
            f"Barren land at {_pct(barren)} signals low organic matter in surrounding soils."
        )

    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} is marginally compatible for Kharif onion.")

    if urban >= 8:
        parts.append(
            f"Urban presence at {_pct(urban)} is a strong commercial signal for onion due to "
            f"direct market access and proximity to cold storage."
        )

    if forest >= 12:
        parts.append(f"Forest cover at {_pct(forest)} is a negative for onion due to high humidity issues.")

    parts.append(
        f"Sulphur fertilisation is non-negotiable for onion: without 45 kg/ha sulphur, "
        f"pungency drops and shelf life falls dramatically."
    )

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for onion (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 68 — Tomato
# ---------------------------------------------------------------------------

def explain_tomato(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    urban  = obs[I_URBAN]
    water  = obs[I_WATER]
    barren = obs[I_BARREN]
    forest = obs[I_FOREST]
    range_ = obs[I_RANGE]

    parts = []

    if agri >= 60:
        parts.append(f"Agricultural land at {_pct(agri)} is a strong base for tomato.")
    elif agri >= 35:
        parts.append(f"Farmland at {_pct(agri)} gives tomato a workable cultivation base.")
    else:
        parts.append(f"Agricultural coverage at {_pct(agri)} is limited. Small-scale polyhouses are better.")

    if urban >= 10:
        parts.append(
            f"Urban coverage at {_pct(urban)} is a significant commercial advantage for tomato, "
            f"reducing logistics cost for this wet market crop."
        )
    elif urban >= 4:
        parts.append(f"Urban presence at {_pct(urban)} provides useful market proximity.")

    if water >= 8:
        parts.append(
            f"Water body coverage at {_pct(water)} is a drainage risk for tomato. "
            f"Tomato roots are extremely sensitive to waterlogging."
        )

    if barren >= 15:
        parts.append(
            f"Barren land at {_pct(barren)} indicates nutritionally limited soils. "
            f"Tomato is a heavy feeder — calcium deficiency causes blossom-end rot."
        )

    if range_ >= 15:
        parts.append(f"Rangeland at {_pct(range_)} is marginally compatible.")

    if 5 <= forest <= 20:
        parts.append(
            f"Forest coverage at {_pct(forest)} may indicate a cooler microclimate, which "
            f"improves tomato fruit quality (lycopene content and firmness)."
        )

    parts.append(
        f"Disease management is the defining factor in tomato profitability: Tuta absoluta "
        f"and Tomato Leaf Curl Virus require proactive prevention from the nursery stage."
    )

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for tomato (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 82 — Mango
# ---------------------------------------------------------------------------

def explain_mango(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water  = obs[I_WATER]
    urban  = obs[I_URBAN]
    forest = obs[I_FOREST]

    parts = []

    if agri >= 55:
        parts.append(f"Agricultural land at {_pct(agri)} is a solid foundation for mango.")
    elif agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} provides a workable base.")
    else:
        parts.append(
            f"Agricultural coverage at {_pct(agri)} is limited, but mango remains a viable recommendation "
            f"as it establishes successfully on marginal land."
        )

    if barren >= 20:
        parts.append(
            f"Barren land at {_pct(barren)} is more manageable for mango than for almost any annual crop — "
            f"its extensive root system colonises sub-soil moisture over years."
        )
    elif barren >= 10:
        parts.append(f"Barren land at {_pct(barren)} is within mango's tolerance.")

    if range_ >= 15:
        parts.append(f"Rangeland at {_pct(range_)} is compatible with mango development.")

    if water >= 10:
        parts.append(
            f"Water body coverage at {_pct(water)} is a root-zone waterlogging alert. "
            f"Mango cannot tolerate a high water table."
        )

    if urban >= 8:
        parts.append(
            f"Urban coverage at {_pct(urban)} is a positive for mango — it is a widely planted "
            f"compound and avenue tree."
        )

    if 8 <= forest <= 20:
        parts.append(
            f"Forest cover at {_pct(forest)} suggests a humid microclimate. Ensure there is "
            f"a 6-week dry spell annually to trigger flowering."
        )

    parts.append(
        f"Alternate bearing is mango's primary productivity challenge — paclobutrazol soil drench "
        f"is the most effective management intervention for consistent yields."
    )

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for mango (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 6 — Finger Millet / Ragi
# ---------------------------------------------------------------------------

def explain_finger_millet(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    range_ = obs[I_RANGE]
    forest = obs[I_FOREST]
    barren = obs[I_BARREN]
    water  = obs[I_WATER]

    parts = []

    if agri >= 45:
        parts.append(
            f"Agricultural land at {_pct(agri)} gives ragi a strong cultivation base, "
            f"especially for rainfed Kharif systems."
        )
    elif agri >= 25:
        parts.append(f"Farmland at {_pct(agri)} is adequate for finger millet.")
    else:
        parts.append(
            f"Agricultural coverage at {_pct(agri)} is limited, but ragi can still perform "
            f"on marginal plots with low external input."
        )

    if range_ >= 25:
        parts.append(
            f"Rangeland at {_pct(range_)} is a positive signal for ragi since it tolerates "
            f"semi-arid upland conditions better than rice or wheat."
        )

    if 8 <= forest <= 25:
        parts.append(
            f"Forest coverage at {_pct(forest)} suggests a cooler upland microclimate, "
            f"which is generally compatible with finger millet."
        )
    elif forest > 25:
        parts.append(f"Forest at {_pct(forest)} may increase shade and reduce ragi productivity.")

    if barren >= 20:
        parts.append(
            f"Barren land at {_pct(barren)} is manageable for ragi due to its hardy root system, "
            f"but nutrient correction with organic matter will still be needed."
        )

    if water >= 10:
        parts.append(f"Water coverage at {_pct(water)} is a drainage caution because ragi is not a wetland crop.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for finger millet (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 17 — Pigeon Pea / Tur (Arhar)
# ---------------------------------------------------------------------------

def explain_pigeon_pea(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water  = obs[I_WATER]
    urban  = obs[I_URBAN]

    parts = []

    if agri >= 50:
        parts.append(f"Agricultural land at {_pct(agri)} is a strong foundation for pigeon pea.")
    elif agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} provides a workable base for tur.")
    else:
        parts.append(f"Agricultural coverage at {_pct(agri)} is low, so yields will depend on plot-level management.")

    if barren >= 18 or range_ >= 22:
        parts.append(
            f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) shares support pigeon pea's "
            f"strength in dryland and low-input conditions."
        )

    if water >= 8:
        parts.append(
            f"Water body coverage at {_pct(water)} is a caution for tur because early-stage "
            f"waterlogging causes root disease and stand loss."
        )

    if urban >= 12:
        parts.append(
            f"Urban presence at {_pct(urban)} can still favor pigeon pea as a peri-urban pulse "
            f"for direct grain market supply."
        )

    parts.append("A practical advantage: deep roots improve soil structure and leave residual nitrogen for the next crop.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for pigeon pea (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 18 — Lentil / Masur
# ---------------------------------------------------------------------------

def explain_lentil(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    barren = obs[I_BARREN]
    water  = obs[I_WATER]
    forest = obs[I_FOREST]
    range_ = obs[I_RANGE]

    parts = []

    if agri >= 55:
        parts.append(f"Farmland at {_pct(agri)} is an excellent base for lentil in the Rabi season.")
    elif agri >= 35:
        parts.append(f"Agricultural coverage at {_pct(agri)} is adequate for masur cultivation.")
    else:
        parts.append(f"Agricultural land at {_pct(agri)} is limited, which reduces lentil's consistency.")

    if barren >= 15:
        parts.append(
            f"Barren coverage at {_pct(barren)} is manageable for lentil, which tolerates relatively "
            f"poor soils better than many winter crops."
        )

    if range_ >= 18:
        parts.append(f"Rangeland at {_pct(range_)} aligns with lentil's dryland adaptation.")

    if water >= 6:
        parts.append(
            f"Water presence at {_pct(water)} is a risk signal because lentil is highly sensitive "
            f"to waterlogging during flowering and pod fill."
        )

    if forest >= 15:
        parts.append(f"Forest cover at {_pct(forest)} may reduce light availability needed for lentil branching.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for lentil (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 30 — Sunflower
# ---------------------------------------------------------------------------

def explain_sunflower(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water  = obs[I_WATER]
    forest = obs[I_FOREST]

    parts = []

    if agri >= 55:
        parts.append(f"Agricultural land at {_pct(agri)} is a strong positive for sunflower.")
    elif agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} gives sunflower a workable production base.")
    else:
        parts.append(f"Agricultural coverage at {_pct(agri)} is a limiting factor for sunflower yields.")

    if barren >= 15:
        parts.append(f"Barren land at {_pct(barren)} is tolerable for sunflower with proper nutrient management.")

    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} supports sunflower's suitability in open dry environments.")

    if water >= 8:
        parts.append(
            f"Water body share at {_pct(water)} is a drainage alert since sunflower roots "
            f"decline rapidly under standing moisture."
        )

    if forest >= 12:
        parts.append(f"Forest cover at {_pct(forest)} can reduce sunflower head development due to shade.")

    parts.append("Pollinator activity is yield-critical in sunflower, so synchronized flowering and bee movement matter.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for sunflower (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 31 — Sesame / Til
# ---------------------------------------------------------------------------

def explain_sesame(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water  = obs[I_WATER]
    urban  = obs[I_URBAN]

    parts = []

    if agri >= 45:
        parts.append(f"Agricultural land at {_pct(agri)} provides a good production base for sesame.")
    elif agri >= 20:
        parts.append(f"Farmland at {_pct(agri)} is sufficient for til under rainfed management.")
    else:
        parts.append(f"Agricultural coverage at {_pct(agri)} is low, but sesame can still fit marginal plots.")

    if barren >= 20:
        parts.append(
            f"Barren land at {_pct(barren)} is not necessarily negative for sesame, "
            f"which performs in light, low-fertility soils where other oilseeds struggle."
        )

    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} is compatible with sesame's drought-tolerant profile.")

    if water >= 6:
        parts.append(
            f"Water body coverage at {_pct(water)} is a key risk for sesame because "
            f"the crop is highly sensitive to waterlogging and root diseases."
        )

    if urban >= 15:
        parts.append(f"Urban share at {_pct(urban)} supports market linkage for high-value sesame seed sales.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for sesame (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 36 — Palm Oil
# ---------------------------------------------------------------------------

def explain_palm_oil(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    forest = obs[I_FOREST]
    water  = obs[I_WATER]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]

    parts = []

    if agri >= 60:
        parts.append(f"Agricultural land at {_pct(agri)} is a strong base for oil palm establishment.")
    elif agri >= 40:
        parts.append(f"Farmland at {_pct(agri)} is workable, but plantation economics improve with larger contiguous blocks.")
    else:
        parts.append(f"Agricultural coverage at {_pct(agri)} is low for a long-duration plantation crop like oil palm.")

    if forest >= 25:
        parts.append(
            f"Forest share at {_pct(forest)} indicates a humid environment compatible with oil palm growth, "
            f"provided legal land-use compliance is ensured."
        )

    if water >= 5:
        parts.append(f"Water presence at {_pct(water)} supports oil palm's high annual moisture demand.")
    else:
        parts.append(f"Water coverage at {_pct(water)} is low for oil palm and will increase irrigation dependence.")

    if barren >= 10:
        parts.append(f"Barren land at {_pct(barren)} is a risk because young palms need fertile, moisture-retentive soils.")

    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} is a weak signal for oil palm due to lower soil development.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for palm oil (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 37 — Coconut
# ---------------------------------------------------------------------------

def explain_coconut(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    forest = obs[I_FOREST]
    water  = obs[I_WATER]
    urban  = obs[I_URBAN]
    barren = obs[I_BARREN]

    parts = []

    if agri >= 50:
        parts.append(f"Agricultural land at {_pct(agri)} supports coconut orchard establishment.")
    elif agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} is adequate for coconut with good pit preparation.")
    else:
        parts.append(f"Agricultural coverage at {_pct(agri)} is limited for commercial coconut blocks.")

    if forest >= 15:
        parts.append(f"Forest coverage at {_pct(forest)} aligns with coconut's preference for humid tropical environments.")

    if water >= 5:
        parts.append(f"Water bodies at {_pct(water)} are a positive moisture signal for coconut palms.")

    if urban >= 10:
        parts.append(
            f"Urban share at {_pct(urban)} is compatible with coconut as a high-value peri-urban and homestead crop."
        )

    if barren >= 12:
        parts.append(f"Barren coverage at {_pct(barren)} suggests nutrient and organic matter correction before planting.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for coconut (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 45 — Tea
# ---------------------------------------------------------------------------

def explain_tea(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    forest = obs[I_FOREST]
    water  = obs[I_WATER]
    range_ = obs[I_RANGE]
    barren = obs[I_BARREN]

    parts = []

    if forest >= 30:
        parts.append(
            f"Forest coverage at {_pct(forest)} is a major positive for tea because it indicates "
            f"humid upland conditions and partial shade systems."
        )
    elif forest >= 15:
        parts.append(f"Forest share at {_pct(forest)} is moderately supportive for tea landscapes.")
    else:
        parts.append(f"Forest coverage at {_pct(forest)} is lower than typical tea ecologies.")

    if agri >= 40:
        parts.append(f"Agricultural land at {_pct(agri)} supports estate-style tea management.")
    else:
        parts.append(f"Farmland at {_pct(agri)} suggests tea may need concentrated block development.")

    if water >= 3:
        parts.append(f"Water presence at {_pct(water)} supports tea's consistent moisture requirement.")

    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} is generally less suitable than shaded hill soils for tea.")

    if barren >= 10:
        parts.append(f"Barren land at {_pct(barren)} is a warning for tea, which is sensitive to shallow degraded soils.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for tea (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 46 — Coffee Arabica
# ---------------------------------------------------------------------------

def explain_coffee_arabica(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    forest = obs[I_FOREST]
    water  = obs[I_WATER]
    urban  = obs[I_URBAN]
    barren = obs[I_BARREN]

    parts = []

    if forest >= 35:
        parts.append(
            f"Forest coverage at {_pct(forest)} strongly supports Arabica coffee, "
            f"which performs best in shaded, biodiverse hill systems."
        )
    elif forest >= 20:
        parts.append(f"Forest share at {_pct(forest)} is reasonably supportive for Arabica.")
    else:
        parts.append(f"Forest coverage at {_pct(forest)} is lower than ideal for Arabica shade management.")

    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} provides a workable managed base for coffee blocks.")

    if water >= 2:
        parts.append(f"Water presence at {_pct(water)} supports flowering and berry development management.")

    if urban >= 10:
        parts.append(f"Urban share at {_pct(urban)} can improve access to labor and processing infrastructure.")

    if barren >= 10:
        parts.append(f"Barren land at {_pct(barren)} is a risk because Arabica is sensitive to moisture stress.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for coffee Arabica (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 52 — Black Pepper
# ---------------------------------------------------------------------------

def explain_black_pepper(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    forest = obs[I_FOREST]
    agri   = obs[I_AGRI]
    water  = obs[I_WATER]
    barren = obs[I_BARREN]
    urban  = obs[I_URBAN]

    parts = []

    if forest >= 30:
        parts.append(
            f"Forest cover at {_pct(forest)} is a key positive for black pepper because "
            f"vines benefit from humid, partially shaded tree-based systems."
        )
    elif forest >= 15:
        parts.append(f"Forest coverage at {_pct(forest)} is moderately supportive for pepper vines.")
    else:
        parts.append(f"Forest at {_pct(forest)} is below preferred conditions for black pepper.")

    if agri >= 35:
        parts.append(f"Agricultural share at {_pct(agri)} supports managed spice cultivation and support-tree integration.")

    if water >= 2:
        parts.append(f"Water presence at {_pct(water)} is beneficial for maintaining root-zone moisture in dry spells.")

    if barren >= 10:
        parts.append(f"Barren land at {_pct(barren)} is a caution because black pepper is shallow-rooted and moisture-sensitive.")

    if urban >= 10:
        parts.append(f"Urban share at {_pct(urban)} can support high-value spice marketing and post-harvest handling.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for black pepper (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 55 — Turmeric
# ---------------------------------------------------------------------------

def explain_turmeric(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    forest = obs[I_FOREST]
    water  = obs[I_WATER]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]

    parts = []

    if agri >= 50:
        parts.append(f"Agricultural land at {_pct(agri)} is a strong base for turmeric cultivation.")
    elif agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} is workable for turmeric with good organic matter.")
    else:
        parts.append(f"Agricultural coverage at {_pct(agri)} is limited for rhizome-intensive turmeric production.")

    if 10 <= forest <= 35:
        parts.append(f"Forest coverage at {_pct(forest)} supports the warm-humid microclimate turmeric prefers.")

    if water >= 4:
        parts.append(f"Water presence at {_pct(water)} supports turmeric's consistent moisture requirement.")
    else:
        parts.append(f"Water coverage at {_pct(water)} indicates irrigation planning will be important for turmeric.")

    if barren >= 12:
        parts.append(f"Barren land at {_pct(barren)} signals lower fertility and higher mulching demand for turmeric.")

    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} is less ideal than cultivated loamy fields for turmeric.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for turmeric (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 57 — Chili / Capsicum
# ---------------------------------------------------------------------------

def explain_chili(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    urban  = obs[I_URBAN]
    water  = obs[I_WATER]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    forest = obs[I_FOREST]

    parts = []

    if agri >= 55:
        parts.append(f"Agricultural land at {_pct(agri)} is a strong production base for chili.")
    elif agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} is adequate for chili with careful fertigation.")
    else:
        parts.append(f"Agricultural share at {_pct(agri)} is low for consistent chili output.")

    if urban >= 8:
        parts.append(f"Urban coverage at {_pct(urban)} is a market advantage for fresh and dry chili trade.")

    if water >= 6:
        parts.append(
            f"Water share at {_pct(water)} is a caution for chili because excess moisture "
            f"increases wilt and fruit rot pressure."
        )

    if barren >= 12:
        parts.append(f"Barren land at {_pct(barren)} indicates nutrient correction is needed for chili quality.")

    if range_ >= 15:
        parts.append(f"Rangeland at {_pct(range_)} is only partially suitable for chili without field development.")

    if forest >= 12:
        parts.append(f"Forest cover at {_pct(forest)} can increase humidity-related disease risk in chili.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for chili (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 83 — Banana
# ---------------------------------------------------------------------------

def explain_banana(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    water  = obs[I_WATER]
    forest = obs[I_FOREST]
    urban  = obs[I_URBAN]
    barren = obs[I_BARREN]

    parts = []

    if agri >= 55:
        parts.append(f"Agricultural land at {_pct(agri)} strongly supports banana plantation planning.")
    elif agri >= 35:
        parts.append(f"Farmland at {_pct(agri)} is workable for banana with drip irrigation.")
    else:
        parts.append(f"Agricultural coverage at {_pct(agri)} is limited for commercial banana systems.")

    if water >= 5:
        parts.append(f"Water presence at {_pct(water)} is a strong positive because banana has high year-round water demand.")
    else:
        parts.append(f"Water coverage at {_pct(water)} is low and will require dependable irrigation infrastructure.")

    if 10 <= forest <= 30:
        parts.append(f"Forest share at {_pct(forest)} suggests humid conditions suitable for banana growth.")

    if urban >= 10:
        parts.append(f"Urban coverage at {_pct(urban)} supports banana's quick market turnaround and transport economics.")

    if barren >= 12:
        parts.append(f"Barren land at {_pct(barren)} is a caution because banana is sensitive to shallow low-fertility soils.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for banana (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 89 — Orange / Citrus
# ---------------------------------------------------------------------------

def explain_orange(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    barren = obs[I_BARREN]
    water  = obs[I_WATER]
    forest = obs[I_FOREST]
    urban  = obs[I_URBAN]

    parts = []

    if agri >= 50:
        parts.append(f"Agricultural land at {_pct(agri)} is a strong base for citrus orchard establishment.")
    elif agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} is workable for orange with good rootstock and drainage choices.")
    else:
        parts.append(f"Agricultural coverage at {_pct(agri)} is limited for stable citrus performance.")

    if 10 <= barren <= 30:
        parts.append(
            f"Barren land at {_pct(barren)} is manageable for citrus if pits are enriched, "
            f"as many orange systems perform on moderately dry soils."
        )
    elif barren > 30:
        parts.append(f"Barren share at {_pct(barren)} is high and raises orchard establishment risk.")

    if water >= 6:
        parts.append(
            f"Water presence at {_pct(water)} supports irrigation availability, but citrus still requires "
            f"strict drainage to avoid Phytophthora root rot."
        )

    if forest >= 15:
        parts.append(f"Forest cover at {_pct(forest)} may increase humidity and disease pressure in citrus canopies.")

    if urban >= 8:
        parts.append(f"Urban share at {_pct(urban)} is beneficial for fresh citrus marketing and value chain access.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for orange/citrus (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 96 — Date Palm
# ---------------------------------------------------------------------------

def explain_date_palm(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri   = obs[I_AGRI]
    barren = obs[I_BARREN]
    water  = obs[I_WATER]
    range_ = obs[I_RANGE]
    forest = obs[I_FOREST]

    parts = []

    if barren >= 35:
        parts.append(
            f"Barren land at {_pct(barren)} is a strong positive for date palm, "
            f"which is specifically adapted to arid and saline-prone landscapes."
        )
    elif barren >= 20:
        parts.append(f"Barren coverage at {_pct(barren)} aligns with date palm's dryland suitability.")
    else:
        parts.append(f"Barren share at {_pct(barren)} is lower than typical date-palm ecologies.")

    if agri >= 25:
        parts.append(f"Agricultural land at {_pct(agri)} supports structured orchard layout and management.")

    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} is compatible with date palm's arid-zone profile.")

    if water >= 3:
        parts.append(
            f"Water presence at {_pct(water)} is useful because date palm needs irrigation despite "
            f"its drought tolerance, especially in establishment years."
        )

    if forest >= 8:
        parts.append(f"Forest cover at {_pct(forest)} is generally not aligned with date palm's hot, open conditions.")

    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for date palm (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 7 — Barley
# ---------------------------------------------------------------------------

def explain_barley(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water = obs[I_WATER]
    parts = []
    if agri >= 50:
        parts.append(f"Agricultural land at {_pct(agri)} is a strong base for barley in cool-season systems.")
    elif agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} is adequate for barley with proper seedbed preparation.")
    else:
        parts.append(f"Agricultural coverage at {_pct(agri)} is low, which can limit barley consistency.")
    if barren >= 15 or range_ >= 20:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) shares fit barley's moderate dryland tolerance.")
    if water >= 8:
        parts.append(f"Water presence at {_pct(water)} is a drainage caution since barley dislikes prolonged waterlogging.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for barley (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 8 — Oats
# ---------------------------------------------------------------------------

def explain_oats(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    forest = obs[I_FOREST]
    water = obs[I_WATER]
    range_ = obs[I_RANGE]
    parts = []
    if agri >= 45:
        parts.append(f"Farmland at {_pct(agri)} supports oats well, especially for fodder and dual-use systems.")
    else:
        parts.append(f"Agricultural share at {_pct(agri)} is modest, so oats performance will be field-specific.")
    if 8 <= forest <= 25:
        parts.append(f"Forest coverage at {_pct(forest)} can indicate a cooler microclimate favorable for oats.")
    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} is compatible with oats in rainfed fodder zones.")
    if water >= 10:
        parts.append(f"Water body share at {_pct(water)} signals potential lodging and root stress in oats.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for oats (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 9 — Rye
# ---------------------------------------------------------------------------

def explain_rye(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    forest = obs[I_FOREST]
    parts = []
    if agri >= 40:
        parts.append(f"Agricultural land at {_pct(agri)} provides a workable base for rye cultivation.")
    else:
        parts.append(f"Farmland at {_pct(agri)} is limited, but rye is more resilient than wheat on marginal soils.")
    if barren >= 18 or range_ >= 20:
        parts.append(f"Dryland signals from barren ({_pct(barren)}) and rangeland ({_pct(range_)}) are acceptable for rye.")
    if forest >= 20:
        parts.append(f"Forest cover at {_pct(forest)} may reduce rye performance due to shading.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for rye (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 10 — Triticale
# ---------------------------------------------------------------------------

def explain_triticale(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water = obs[I_WATER]
    parts = []
    if agri >= 45:
        parts.append(f"Farmland at {_pct(agri)} is a strong base for triticale in grain-plus-fodder systems.")
    else:
        parts.append(f"Agricultural coverage at {_pct(agri)} is moderate, where triticale can still outperform wheat under stress.")
    if barren >= 15 or range_ >= 18:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) proportions align with triticale's stress tolerance.")
    if water >= 10:
        parts.append(f"Water presence at {_pct(water)} is a caution because triticale still needs good root-zone aeration.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for triticale (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 11 — Teff
# ---------------------------------------------------------------------------

def explain_teff(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    range_ = obs[I_RANGE]
    barren = obs[I_BARREN]
    water = obs[I_WATER]
    parts = []
    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} gives teff a workable base, especially under low-input systems.")
    if range_ >= 25:
        parts.append(f"Rangeland at {_pct(range_)} is a positive for teff's adaptation to semi-arid open environments.")
    if barren >= 20:
        parts.append(f"Barren share at {_pct(barren)} is manageable for teff with organic matter support.")
    if water >= 8:
        parts.append(f"Water coverage at {_pct(water)} can increase lodging risk in teff if drainage is poor.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for teff (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 12 — Foxtail Millet / Kangni
# ---------------------------------------------------------------------------

def explain_foxtail_millet(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water = obs[I_WATER]
    parts = []
    if agri >= 35:
        parts.append(f"Farmland at {_pct(agri)} supports foxtail millet under short-duration rainfed cycles.")
    if barren >= 18 or range_ >= 20:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) profiles suit foxtail millet's drought resilience.")
    if water >= 8:
        parts.append(f"Water share at {_pct(water)} is a caution because millet roots are sensitive to standing water.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for foxtail millet (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 13 — Kodo Millet
# ---------------------------------------------------------------------------

def explain_kodo_millet(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    forest = obs[I_FOREST]
    range_ = obs[I_RANGE]
    parts = []
    if agri >= 30:
        parts.append(f"Agricultural share at {_pct(agri)} is enough for kodo millet in low-input cultivation.")
    if barren >= 15:
        parts.append(f"Barren land at {_pct(barren)} is compatible with kodo millet's hardiness.")
    if 10 <= forest <= 25:
        parts.append(f"Forest cover at {_pct(forest)} can support the mixed upland ecology where kodo often performs well.")
    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} further supports kodo millet suitability.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for kodo millet (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 14 — Little Millet / Kutki
# ---------------------------------------------------------------------------

def explain_little_millet(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    forest = obs[I_FOREST]
    range_ = obs[I_RANGE]
    water = obs[I_WATER]
    parts = []
    if agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} provides a practical base for little millet cultivation.")
    if barren >= 18 or range_ >= 20:
        parts.append(f"Dryland indicators from barren ({_pct(barren)}) and rangeland ({_pct(range_)}) fit little millet.")
    if forest >= 12:
        parts.append(f"Forest share at {_pct(forest)} is acceptable for little millet in mixed rainfed landscapes.")
    if water >= 8:
        parts.append(f"Water coverage at {_pct(water)} is a drainage caution for this upland millet.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for little millet (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 15 — Proso Millet
# ---------------------------------------------------------------------------

def explain_proso_millet(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water = obs[I_WATER]
    parts = []
    if agri >= 30:
        parts.append(f"Agricultural land at {_pct(agri)} is suitable for proso millet's short-cycle production.")
    if barren >= 20:
        parts.append(f"Barren land at {_pct(barren)} is manageable for proso millet with minimal input demand.")
    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} aligns with proso millet's dryland adaptation.")
    if water >= 8:
        parts.append(f"Water share at {_pct(water)} may increase risk of stand loss in poorly drained pockets.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for proso millet (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 19 — Green Gram / Moong
# ---------------------------------------------------------------------------

def explain_green_gram(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    water = obs[I_WATER]
    range_ = obs[I_RANGE]
    parts = []
    if agri >= 45:
        parts.append(f"Farmland at {_pct(agri)} strongly supports moong as a short-duration pulse crop.")
    elif agri >= 25:
        parts.append(f"Agricultural coverage at {_pct(agri)} is adequate for moong with good weed control.")
    else:
        parts.append(f"Agricultural share at {_pct(agri)} is low, reducing moong productivity potential.")
    if barren >= 15 or range_ >= 18:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) are manageable for moong under rainfed conditions.")
    if water >= 8:
        parts.append(f"Water presence at {_pct(water)} is a caution because moong is sensitive to waterlogging.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for green gram/moong (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 20 — Black Gram / Urad
# ---------------------------------------------------------------------------

def explain_black_gram(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    water = obs[I_WATER]
    range_ = obs[I_RANGE]
    parts = []
    if agri >= 40:
        parts.append(f"Farmland at {_pct(agri)} gives urad a workable cultivation base.")
    else:
        parts.append(f"Agricultural land at {_pct(agri)} is limited for stable urad output.")
    if barren >= 15 or range_ >= 18:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) profiles are broadly compatible with urad.")
    if water >= 8:
        parts.append(f"Water coverage at {_pct(water)} raises root-rot risk in black gram.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for black gram/urad (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 21 — Kidney Bean / Rajma
# ---------------------------------------------------------------------------

def explain_kidney_bean(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    forest = obs[I_FOREST]
    water = obs[I_WATER]
    range_ = obs[I_RANGE]
    parts = []
    if agri >= 40:
        parts.append(f"Agricultural land at {_pct(agri)} supports rajma production with proper nutrition management.")
    if forest >= 12:
        parts.append(f"Forest coverage at {_pct(forest)} may indicate cooler conditions that can favor rajma in suitable elevations.")
    if range_ >= 15:
        parts.append(f"Rangeland at {_pct(range_)} is partially suitable where soils are improved before planting.")
    if water >= 8:
        parts.append(f"Water share at {_pct(water)} is a caution because rajma is vulnerable to waterlogging and disease.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for kidney bean/rajma (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 22 — Cowpea / Lobia
# ---------------------------------------------------------------------------

def explain_cowpea(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water = obs[I_WATER]
    parts = []
    if agri >= 35:
        parts.append(f"Farmland at {_pct(agri)} is adequate for cowpea in grain, fodder, or vegetable mode.")
    if barren >= 20 or range_ >= 22:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) shares align with cowpea's drought tolerance.")
    if water >= 8:
        parts.append(f"Water presence at {_pct(water)} is a caution for cowpea due to stem and root disease pressure.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for cowpea/lobia (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 23 — Field Pea
# ---------------------------------------------------------------------------

def explain_field_pea(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    water = obs[I_WATER]
    barren = obs[I_BARREN]
    forest = obs[I_FOREST]
    parts = []
    if agri >= 45:
        parts.append(f"Agricultural land at {_pct(agri)} is a strong base for field pea in winter rotations.")
    else:
        parts.append(f"Farmland at {_pct(agri)} is moderate, so field pea should be focused on better-drained plots.")
    if barren >= 12:
        parts.append(f"Barren share at {_pct(barren)} is manageable for pea with phosphorus and organic matter support.")
    if water >= 8:
        parts.append(f"Water coverage at {_pct(water)} can increase lodging and root disease in field pea.")
    if forest >= 15:
        parts.append(f"Forest at {_pct(forest)} may reduce sunlight needed for robust pod set.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for field pea (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 25 — Moth Bean
# ---------------------------------------------------------------------------

def explain_moth_bean(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    agri = obs[I_AGRI]
    water = obs[I_WATER]
    parts = []
    if barren >= 30:
        parts.append(f"Barren land at {_pct(barren)} is a major positive for moth bean in arid farming systems.")
    if range_ >= 30:
        parts.append(f"Rangeland at {_pct(range_)} also supports moth bean's extreme dryland adaptability.")
    if agri >= 20:
        parts.append(f"Farmland at {_pct(agri)} helps improve establishment and management despite harsh conditions.")
    if water >= 6:
        parts.append(f"Water share at {_pct(water)} can be a risk because moth bean performs best under low-moisture regimes.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for moth bean (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 26 — Horse Gram / Kulthi
# ---------------------------------------------------------------------------

def explain_horse_gram(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water = obs[I_WATER]
    forest = obs[I_FOREST]
    parts = []
    if agri >= 30:
        parts.append(f"Agricultural land at {_pct(agri)} gives horse gram a workable low-input production base.")
    if barren >= 20 or range_ >= 25:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) are favorable for horse gram's drought hardiness.")
    if water >= 8:
        parts.append(f"Water body share at {_pct(water)} is a caution because horse gram performs poorly in waterlogged soils.")
    if forest >= 20:
        parts.append(f"Forest at {_pct(forest)} may reduce sunlight and suppress horse gram yields.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for horse gram (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 27 — Cluster Bean / Guar
# ---------------------------------------------------------------------------

def explain_cluster_bean(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water = obs[I_WATER]
    parts = []
    if agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} supports guar cultivation and management access.")
    if barren >= 25 or range_ >= 30:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) strongly favor cluster bean in arid conditions.")
    if water >= 6:
        parts.append(f"Water coverage at {_pct(water)} is a risk for guar because excess moisture harms root health.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for cluster bean/guar (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 32 — Linseed / Flax
# ---------------------------------------------------------------------------

def explain_linseed(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water = obs[I_WATER]
    parts = []
    if agri >= 40:
        parts.append(f"Agricultural land at {_pct(agri)} is a good base for linseed/flax production.")
    else:
        parts.append(f"Farmland at {_pct(agri)} is moderate, so linseed should target the best-drained fields.")
    if barren >= 15 or range_ >= 18:
        parts.append(f"Dryland signals from barren ({_pct(barren)}) and rangeland ({_pct(range_)}) are compatible with linseed.")
    if water >= 8:
        parts.append(f"Water share at {_pct(water)} is a caution due to lodging and fungal risk.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for linseed/flax (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 33 — Castor
# ---------------------------------------------------------------------------

def explain_castor(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water = obs[I_WATER]
    parts = []
    if agri >= 35:
        parts.append(f"Farmland at {_pct(agri)} supports castor as a hardy industrial oilseed crop.")
    if barren >= 25:
        parts.append(f"Barren land at {_pct(barren)} is favorable for castor's drought tolerance.")
    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} also aligns with castor's dry-zone suitability.")
    if water >= 8:
        parts.append(f"Water presence at {_pct(water)} is a caution as castor roots are vulnerable under prolonged wetness.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for castor (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 34 — Safflower
# ---------------------------------------------------------------------------

def explain_safflower(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water = obs[I_WATER]
    parts = []
    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} provides a workable base for safflower.")
    if barren >= 20 or range_ >= 25:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) fit safflower's low-moisture adaptation.")
    if water >= 8:
        parts.append(f"Water coverage at {_pct(water)} is a caution because safflower performs best in dry post-rainy seasons.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for safflower (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 35 — Niger / Ramtil
# ---------------------------------------------------------------------------

def explain_niger(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    forest = obs[I_FOREST]
    range_ = obs[I_RANGE]
    water = obs[I_WATER]
    parts = []
    if agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} is adequate for niger in low-input hill and tribal farming systems.")
    if barren >= 15 or range_ >= 20:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) are compatible with niger's hardy nature.")
    if forest >= 12:
        parts.append(f"Forest cover at {_pct(forest)} can suit niger in upland mixed landscapes.")
    if water >= 8:
        parts.append(f"Water share at {_pct(water)} may reduce niger performance where drainage is poor.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for niger/ramtil (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 39 — Cotton (Desi)
# ---------------------------------------------------------------------------

def explain_cotton_desi(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    water = obs[I_WATER]
    range_ = obs[I_RANGE]
    parts = []
    if agri >= 45:
        parts.append(f"Agricultural land at {_pct(agri)} is a solid base for desi cotton.")
    if 10 <= barren <= 25:
        parts.append(f"Barren land at {_pct(barren)} is within the tolerance range of desi cotton systems.")
    elif barren > 25:
        parts.append(f"Barren share at {_pct(barren)} is high and may suppress boll set without soil correction.")
    if range_ >= 15:
        parts.append(f"Rangeland at {_pct(range_)} is partly compatible for expansion into hardy cotton blocks.")
    if water >= 10:
        parts.append(f"Water coverage at {_pct(water)} raises risk of boll rot and root stress.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for desi cotton (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 41 — Hemp
# ---------------------------------------------------------------------------

def explain_hemp(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    forest = obs[I_FOREST]
    water = obs[I_WATER]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    parts = []
    if agri >= 40:
        parts.append(f"Farmland at {_pct(agri)} supports hemp establishment for fiber or seed purposes.")
    if forest >= 12:
        parts.append(f"Forest cover at {_pct(forest)} can indicate cooler conditions that support hemp vegetative growth.")
    if range_ >= 15:
        parts.append(f"Rangeland at {_pct(range_)} is partially suitable when converted to prepared fields.")
    if barren >= 15:
        parts.append(f"Barren share at {_pct(barren)} is a caution because hemp performs best in fertile, well-structured soils.")
    if water >= 8:
        parts.append(f"Water presence at {_pct(water)} requires drainage planning to prevent root diseases.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for hemp (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 42 — Kenaf
# ---------------------------------------------------------------------------

def explain_kenaf(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    forest = obs[I_FOREST]
    water = obs[I_WATER]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    parts = []
    if agri >= 40:
        parts.append(f"Agricultural land at {_pct(agri)} is a practical base for kenaf fiber cultivation.")
    if 8 <= forest <= 20:
        parts.append(f"Forest share at {_pct(forest)} suggests a warm-humid profile that can suit kenaf growth.")
    if water >= 4:
        parts.append(f"Water presence at {_pct(water)} supports kenaf biomass development when drainage is controlled.")
    if barren >= 15:
        parts.append(f"Barren land at {_pct(barren)} is a moderate risk for kenaf fiber quality.")
    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} is only partially suitable until soils are improved.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for kenaf (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 44 — Sugar Beet
# ---------------------------------------------------------------------------

def explain_sugar_beet(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    water = obs[I_WATER]
    barren = obs[I_BARREN]
    forest = obs[I_FOREST]
    parts = []
    if agri >= 55:
        parts.append(f"Agricultural land at {_pct(agri)} is a strong prerequisite for sugar beet cultivation.")
    elif agri >= 35:
        parts.append(f"Farmland at {_pct(agri)} is workable for sugar beet with precise nutrient management.")
    else:
        parts.append(f"Agricultural coverage at {_pct(agri)} is low for a high-management root crop like sugar beet.")
    if water >= 6:
        parts.append(f"Water presence at {_pct(water)} supports irrigation needs, but drainage must be maintained.")
    if barren >= 12:
        parts.append(f"Barren share at {_pct(barren)} indicates possible salinity risk for sugar beet quality.")
    if forest >= 15:
        parts.append(f"Forest cover at {_pct(forest)} may reduce light and field uniformity needed for beet yields.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for sugar beet (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 47 — Coffee Robusta
# ---------------------------------------------------------------------------

def explain_coffee_robusta(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    forest = obs[I_FOREST]
    water = obs[I_WATER]
    urban = obs[I_URBAN]
    barren = obs[I_BARREN]
    parts = []
    if forest >= 25:
        parts.append(f"Forest coverage at {_pct(forest)} strongly supports Robusta coffee's humid shaded requirements.")
    elif forest >= 15:
        parts.append(f"Forest share at {_pct(forest)} is moderately supportive for Robusta systems.")
    else:
        parts.append(f"Forest at {_pct(forest)} is lower than preferred for stable Robusta performance.")
    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} supports managed coffee blocks and input logistics.")
    if water >= 2:
        parts.append(f"Water presence at {_pct(water)} helps maintain moisture for flowering and berry fill.")
    if urban >= 10:
        parts.append(f"Urban share at {_pct(urban)} can improve processing and labor access.")
    if barren >= 10:
        parts.append(f"Barren land at {_pct(barren)} is a caution because Robusta favors deeper moisture-retentive soils.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for coffee Robusta (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 48 — Rubber
# ---------------------------------------------------------------------------

def explain_rubber(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    forest = obs[I_FOREST]
    water = obs[I_WATER]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    parts = []
    if agri >= 45:
        parts.append(f"Agricultural land at {_pct(agri)} is a solid base for rubber plantation establishment.")
    if forest >= 30:
        parts.append(f"Forest coverage at {_pct(forest)} signals humid conditions suitable for rubber growth.")
    elif forest >= 15:
        parts.append(f"Forest share at {_pct(forest)} is moderately supportive for rubber.")
    if water >= 2:
        parts.append(f"Water presence at {_pct(water)} supports moisture demand during early plantation years.")
    if barren >= 12:
        parts.append(f"Barren share at {_pct(barren)} is a risk for long-term rubber productivity.")
    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} is less ideal unless converted to structured plantation blocks.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for rubber (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 49 — Tobacco
# ---------------------------------------------------------------------------

def explain_tobacco(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    barren = obs[I_BARREN]
    range_ = obs[I_RANGE]
    water = obs[I_WATER]
    forest = obs[I_FOREST]
    parts = []
    if agri >= 45:
        parts.append(f"Agricultural land at {_pct(agri)} is a strong production base for tobacco.")
    elif agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} is adequate for tobacco with careful curing-chain planning.")
    else:
        parts.append(f"Agricultural share at {_pct(agri)} is limited for profitable tobacco systems.")
    if barren >= 12 or range_ >= 15:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) can support tobacco on lighter soils.")
    if water >= 8:
        parts.append(f"Water presence at {_pct(water)} is a caution due to leaf disease pressure in humid conditions.")
    if forest >= 15:
        parts.append(f"Forest coverage at {_pct(forest)} may increase humidity and curing-quality risk.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for tobacco (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 50 — Cocoa
# ---------------------------------------------------------------------------

def explain_cocoa(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]
    forest = obs[I_FOREST]
    water = obs[I_WATER]
    barren = obs[I_BARREN]
    urban = obs[I_URBAN]
    parts = []
    if forest >= 35:
        parts.append(f"Forest share at {_pct(forest)} is a strong positive for cocoa's shade-loving ecology.")
    elif forest >= 20:
        parts.append(f"Forest coverage at {_pct(forest)} is supportive for cocoa under agroforestry layouts.")
    else:
        parts.append(f"Forest at {_pct(forest)} is below typical cocoa preference for humid shaded conditions.")
    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} supports managed cocoa blocks and intercrop systems.")
    if water >= 2:
        parts.append(f"Water presence at {_pct(water)} supports cocoa's year-round moisture requirement.")
    if barren >= 8:
        parts.append(f"Barren land at {_pct(barren)} is a caution because cocoa is sensitive to drought stress.")
    if urban >= 8:
        parts.append(f"Urban share at {_pct(urban)} can improve access to labor and post-harvest handling.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for cocoa (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 51 — Cashew
# ---------------------------------------------------------------------------

def explain_cashew(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; barren = obs[I_BARREN]; forest = obs[I_FOREST]; range_ = obs[I_RANGE]
    parts = []
    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} supports cashew orchard establishment.")
    if barren >= 18:
        parts.append(f"Barren land at {_pct(barren)} is often manageable for cashew with pit enrichment and mulching.")
    if 10 <= forest <= 30:
        parts.append(f"Forest coverage at {_pct(forest)} indicates a humid-tropical profile favorable for cashew growth.")
    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} is partially suitable once converted to structured orchard blocks.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for cashew (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 53 — Cardamom (Large)
# ---------------------------------------------------------------------------

def explain_cardamom_large(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    forest = obs[I_FOREST]; agri = obs[I_AGRI]; water = obs[I_WATER]; barren = obs[I_BARREN]
    parts = []
    if forest >= 35:
        parts.append(f"Forest cover at {_pct(forest)} is a major positive for large cardamom's shaded hill ecology.")
    else:
        parts.append(f"Forest at {_pct(forest)} is lower than ideal for large cardamom systems.")
    if agri >= 25:
        parts.append(f"Agricultural share at {_pct(agri)} supports managed spice plots and input access.")
    if water >= 2:
        parts.append(f"Water presence at {_pct(water)} supports moisture demand in cardamom root zones.")
    if barren >= 8:
        parts.append(f"Barren land at {_pct(barren)} is a caution due to moisture stress risk in spice crops.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for large cardamom (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 54 — Cardamom (Small/Green)
# ---------------------------------------------------------------------------

def explain_cardamom_small(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    forest = obs[I_FOREST]; agri = obs[I_AGRI]; water = obs[I_WATER]; urban = obs[I_URBAN]
    parts = []
    if forest >= 30:
        parts.append(f"Forest share at {_pct(forest)} strongly supports green cardamom's shade requirement.")
    else:
        parts.append(f"Forest coverage at {_pct(forest)} is moderate and may need managed shade trees.")
    if agri >= 25:
        parts.append(f"Farmland at {_pct(agri)} helps establish plantation-style cardamom management.")
    if water >= 2:
        parts.append(f"Water presence at {_pct(water)} supports stable humidity and irrigation buffering.")
    if urban >= 10:
        parts.append(f"Urban share at {_pct(urban)} can improve access to labor and post-harvest spice handling.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for small cardamom (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 56 — Ginger
# ---------------------------------------------------------------------------

def explain_ginger(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; forest = obs[I_FOREST]; water = obs[I_WATER]; barren = obs[I_BARREN]
    parts = []
    if agri >= 40:
        parts.append(f"Agricultural land at {_pct(agri)} provides a strong base for ginger's rhizome-intensive cultivation.")
    if 10 <= forest <= 35:
        parts.append(f"Forest coverage at {_pct(forest)} supports the warm-humid microclimate ginger prefers.")
    if water >= 3:
        parts.append(f"Water presence at {_pct(water)} helps meet ginger's steady moisture requirement.")
    if barren >= 12:
        parts.append(f"Barren land at {_pct(barren)} signals higher organic-matter and mulching needs for ginger.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for ginger (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 58 — Coriander / Dhania
# ---------------------------------------------------------------------------

def explain_coriander(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; barren = obs[I_BARREN]; range_ = obs[I_RANGE]; water = obs[I_WATER]
    parts = []
    if agri >= 45:
        parts.append(f"Farmland at {_pct(agri)} is a good base for coriander in seed and leaf markets.")
    if barren >= 15 or range_ >= 18:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) are manageable for coriander under dry-season production.")
    if water >= 8:
        parts.append(f"Water share at {_pct(water)} is a caution for coriander due to wilt and blight risk.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for coriander (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 59 — Cumin / Jeera
# ---------------------------------------------------------------------------

def explain_cumin(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; barren = obs[I_BARREN]; range_ = obs[I_RANGE]; water = obs[I_WATER]
    parts = []
    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} supports cumin under cool, dry-season management.")
    if barren >= 20 or range_ >= 20:
        parts.append(f"Dryland signals from barren ({_pct(barren)}) and rangeland ({_pct(range_)}) fit cumin cultivation.")
    if water >= 6:
        parts.append(f"Water presence at {_pct(water)} is a disease risk for cumin, which prefers dry canopy conditions.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for cumin (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 60 — Fenugreek / Methi
# ---------------------------------------------------------------------------

def explain_fenugreek(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; barren = obs[I_BARREN]; range_ = obs[I_RANGE]; water = obs[I_WATER]
    parts = []
    if agri >= 40:
        parts.append(f"Farmland at {_pct(agri)} provides a solid base for fenugreek as leaf and seed crop.")
    if barren >= 15 or range_ >= 18:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) are workable for methi with modest inputs.")
    if water >= 8:
        parts.append(f"Water share at {_pct(water)} is a caution since fenugreek needs moisture but not waterlogging.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for fenugreek (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 61 — Clove
# ---------------------------------------------------------------------------

def explain_clove(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    forest = obs[I_FOREST]; agri = obs[I_AGRI]; water = obs[I_WATER]; barren = obs[I_BARREN]
    parts = []
    if forest >= 35:
        parts.append(f"Forest cover at {_pct(forest)} is strongly favorable for clove's humid evergreen ecology.")
    else:
        parts.append(f"Forest at {_pct(forest)} is lower than typical clove landscapes.")
    if agri >= 25:
        parts.append(f"Agricultural share at {_pct(agri)} supports orchard-style clove management.")
    if water >= 2:
        parts.append(f"Water presence at {_pct(water)} supports moisture stability for clove growth.")
    if barren >= 8:
        parts.append(f"Barren land at {_pct(barren)} is a caution because clove is sensitive to prolonged dry stress.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for clove (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 62 — Nutmeg
# ---------------------------------------------------------------------------

def explain_nutmeg(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    forest = obs[I_FOREST]; agri = obs[I_AGRI]; water = obs[I_WATER]; urban = obs[I_URBAN]
    parts = []
    if forest >= 30:
        parts.append(f"Forest share at {_pct(forest)} supports nutmeg's humid tropical requirement.")
    if agri >= 25:
        parts.append(f"Farmland at {_pct(agri)} enables structured nutmeg orchard management.")
    if water >= 2:
        parts.append(f"Water presence at {_pct(water)} supports moisture availability for long-duration nutmeg trees.")
    if urban >= 8:
        parts.append(f"Urban coverage at {_pct(urban)} can support labor access and value-chain handling for spice crops.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for nutmeg (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 63 — Vanilla
# ---------------------------------------------------------------------------

def explain_vanilla(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    forest = obs[I_FOREST]; agri = obs[I_AGRI]; water = obs[I_WATER]; barren = obs[I_BARREN]
    parts = []
    if forest >= 35:
        parts.append(f"Forest coverage at {_pct(forest)} is a key positive for vanilla's shade and humidity needs.")
    else:
        parts.append(f"Forest at {_pct(forest)} is moderate, so support trees and microclimate management will matter.")
    if agri >= 20:
        parts.append(f"Agricultural share at {_pct(agri)} supports managed vine systems and hand-pollination logistics.")
    if water >= 2:
        parts.append(f"Water presence at {_pct(water)} supports moisture consistency essential for vanilla vines.")
    if barren >= 8:
        parts.append(f"Barren land at {_pct(barren)} is a caution because vanilla is sensitive to drought and root stress.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for vanilla (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 64 — Saffron
# ---------------------------------------------------------------------------

def explain_saffron(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    barren = obs[I_BARREN]; agri = obs[I_AGRI]; range_ = obs[I_RANGE]; water = obs[I_WATER]
    parts = []
    if barren >= 20:
        parts.append(f"Barren land at {_pct(barren)} can be suitable for saffron in well-drained upland soils.")
    if agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} supports careful corm management needed for saffron.")
    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} indicates open dry landscapes compatible with saffron ecology.")
    if water >= 6:
        parts.append(f"Water share at {_pct(water)} is a caution because saffron requires moisture control and no waterlogging.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for saffron (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 65 — Fennel / Saunf
# ---------------------------------------------------------------------------

def explain_fennel(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; barren = obs[I_BARREN]; range_ = obs[I_RANGE]; water = obs[I_WATER]
    parts = []
    if agri >= 40:
        parts.append(f"Agricultural land at {_pct(agri)} is a good base for fennel seed production.")
    if barren >= 15 or range_ >= 20:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) are manageable for fennel with balanced nutrition.")
    if water >= 8:
        parts.append(f"Water coverage at {_pct(water)} is a caution because fennel quality declines in prolonged wetness.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for fennel (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 69 — Brinjal / Eggplant
# ---------------------------------------------------------------------------

def explain_brinjal(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; urban = obs[I_URBAN]; water = obs[I_WATER]; barren = obs[I_BARREN]
    parts = []
    if agri >= 45:
        parts.append(f"Farmland at {_pct(agri)} is a strong base for brinjal cultivation.")
    if urban >= 8:
        parts.append(f"Urban share at {_pct(urban)} supports quick vegetable market access for brinjal.")
    if water >= 8:
        parts.append(f"Water presence at {_pct(water)} is a caution for wilt and root disease risk in brinjal.")
    if barren >= 12:
        parts.append(f"Barren land at {_pct(barren)} indicates nutrient correction needs for sustained harvest.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for brinjal/eggplant (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 70 — Cabbage
# ---------------------------------------------------------------------------

def explain_cabbage(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; water = obs[I_WATER]; forest = obs[I_FOREST]; urban = obs[I_URBAN]
    parts = []
    if agri >= 45:
        parts.append(f"Agricultural land at {_pct(agri)} supports cabbage production under structured nutrient management.")
    if water >= 5:
        parts.append(f"Water share at {_pct(water)} helps irrigation availability, but drainage must stay strict.")
    if 8 <= forest <= 25:
        parts.append(f"Forest cover at {_pct(forest)} may indicate a cooler microclimate favorable for cabbage heads.")
    if urban >= 8:
        parts.append(f"Urban coverage at {_pct(urban)} supports fresh-market supply for cabbage.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for cabbage (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 71 — Cauliflower
# ---------------------------------------------------------------------------

def explain_cauliflower(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; water = obs[I_WATER]; forest = obs[I_FOREST]; urban = obs[I_URBAN]
    parts = []
    if agri >= 45:
        parts.append(f"Farmland at {_pct(agri)} is a strong base for cauliflower cultivation.")
    if water >= 5:
        parts.append(f"Water presence at {_pct(water)} supports irrigation, but excessive moisture can reduce curd quality.")
    if 8 <= forest <= 25:
        parts.append(f"Forest share at {_pct(forest)} suggests a cooler setting suitable for cauliflower curd development.")
    if urban >= 8:
        parts.append(f"Urban share at {_pct(urban)} improves commercialization for this perishable crop.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for cauliflower (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 72 — Garlic
# ---------------------------------------------------------------------------

def explain_garlic(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; barren = obs[I_BARREN]; water = obs[I_WATER]; urban = obs[I_URBAN]
    parts = []
    if agri >= 45:
        parts.append(f"Agricultural land at {_pct(agri)} supports garlic under precise fertility and irrigation control.")
    if barren >= 12:
        parts.append(f"Barren share at {_pct(barren)} indicates the need for organic matter and sulphur correction.")
    if water >= 8:
        parts.append(f"Water coverage at {_pct(water)} is a caution because garlic bulbs are sensitive to excess soil moisture.")
    if urban >= 8:
        parts.append(f"Urban coverage at {_pct(urban)} can support faster bulb marketing and storage movement.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for garlic (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 73 — Okra / Lady's Finger
# ---------------------------------------------------------------------------

def explain_okra(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; urban = obs[I_URBAN]; water = obs[I_WATER]; barren = obs[I_BARREN]
    parts = []
    if agri >= 40:
        parts.append(f"Farmland at {_pct(agri)} gives okra a suitable production base.")
    if urban >= 8:
        parts.append(f"Urban share at {_pct(urban)} is favorable for frequent okra harvest marketing.")
    if water >= 8:
        parts.append(f"Water presence at {_pct(water)} is a caution due to root and stem disease risk in okra.")
    if barren >= 12:
        parts.append(f"Barren land at {_pct(barren)} indicates nutrient constraints that may reduce pod quality.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for okra (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 74 — Pumpkin / Kaddu
# ---------------------------------------------------------------------------

def explain_pumpkin(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; range_ = obs[I_RANGE]; water = obs[I_WATER]; urban = obs[I_URBAN]
    parts = []
    if agri >= 40:
        parts.append(f"Agricultural land at {_pct(agri)} supports pumpkin vine cultivation.")
    if range_ >= 15:
        parts.append(f"Rangeland at {_pct(range_)} is partially suitable after field preparation for vines.")
    if water >= 6:
        parts.append(f"Water presence at {_pct(water)} helps moisture supply, though stagnant wetness should be avoided.")
    if urban >= 8:
        parts.append(f"Urban share at {_pct(urban)} supports short supply-chain sales for pumpkin.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for pumpkin (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 75 — Cucumber
# ---------------------------------------------------------------------------

def explain_cucumber(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; urban = obs[I_URBAN]; water = obs[I_WATER]; forest = obs[I_FOREST]
    parts = []
    if agri >= 40:
        parts.append(f"Farmland at {_pct(agri)} is a workable base for cucumber with trellis and fertigation systems.")
    if urban >= 8:
        parts.append(f"Urban coverage at {_pct(urban)} is a strong positive for cucumber's fresh market demand.")
    if water >= 6:
        parts.append(f"Water share at {_pct(water)} supports irrigation access, but drainage must remain strong.")
    if forest >= 15:
        parts.append(f"Forest at {_pct(forest)} may increase humidity-related mildew pressure in cucumber.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for cucumber (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 76 — Cassava / Tapioca
# ---------------------------------------------------------------------------

def explain_cassava(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; barren = obs[I_BARREN]; forest = obs[I_FOREST]; range_ = obs[I_RANGE]
    parts = []
    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} supports cassava establishment and management.")
    if barren >= 20 or range_ >= 20:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) are often suitable for cassava's hardy profile.")
    if forest >= 15:
        parts.append(f"Forest cover at {_pct(forest)} can support a humid setting favorable for cassava biomass growth.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for cassava/tapioca (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 77 — Sweet Potato
# ---------------------------------------------------------------------------

def explain_sweet_potato(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; barren = obs[I_BARREN]; water = obs[I_WATER]; forest = obs[I_FOREST]
    parts = []
    if agri >= 35:
        parts.append(f"Farmland at {_pct(agri)} gives sweet potato a strong cultivation base.")
    if barren >= 15:
        parts.append(f"Barren share at {_pct(barren)} is manageable for sweet potato with ridge preparation.")
    if water >= 6:
        parts.append(f"Water presence at {_pct(water)} supports moisture but requires drainage to prevent tuber rot.")
    if forest >= 12:
        parts.append(f"Forest cover at {_pct(forest)} may indicate humid conditions suitable for vine growth.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for sweet potato (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 78 — Yam
# ---------------------------------------------------------------------------

def explain_yam(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; forest = obs[I_FOREST]; water = obs[I_WATER]; barren = obs[I_BARREN]
    parts = []
    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} supports yam mound or ridge systems.")
    if forest >= 20:
        parts.append(f"Forest share at {_pct(forest)} supports yam's humid, partially shaded tropical profile.")
    if water >= 3:
        parts.append(f"Water presence at {_pct(water)} helps meet yam moisture needs during tuber bulking.")
    if barren >= 12:
        parts.append(f"Barren land at {_pct(barren)} can restrict yam size unless soil fertility is improved.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for yam (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 79 — Taro / Colocasia
# ---------------------------------------------------------------------------

def explain_taro(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; water = obs[I_WATER]; forest = obs[I_FOREST]; range_ = obs[I_RANGE]
    parts = []
    if agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} provides a workable base for taro cultivation.")
    if water >= 8:
        parts.append(f"Water share at {_pct(water)} is a positive for taro, which tolerates higher soil moisture than many tubers.")
    if forest >= 15:
        parts.append(f"Forest coverage at {_pct(forest)} can support humid conditions favorable for colocasia.")
    if range_ >= 20:
        parts.append(f"Rangeland at {_pct(range_)} is less suitable unless converted to moist cultivated beds.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for taro/colocasia (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 80 — Spinach
# ---------------------------------------------------------------------------

def explain_spinach(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; urban = obs[I_URBAN]; water = obs[I_WATER]; forest = obs[I_FOREST]
    parts = []
    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} supports high-frequency spinach harvest cycles.")
    if urban >= 10:
        parts.append(f"Urban coverage at {_pct(urban)} is a strong commercial advantage for leafy vegetable turnover.")
    if water >= 4:
        parts.append(f"Water presence at {_pct(water)} supports irrigation for continuous spinach production.")
    if forest >= 12:
        parts.append(f"Forest share at {_pct(forest)} may indicate cooler conditions that can reduce bolting risk.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for spinach (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 81 — Carrot
# ---------------------------------------------------------------------------

def explain_carrot(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; barren = obs[I_BARREN]; water = obs[I_WATER]; forest = obs[I_FOREST]
    parts = []
    if agri >= 40:
        parts.append(f"Farmland at {_pct(agri)} is a good base for carrot production.")
    if barren >= 12:
        parts.append(f"Barren share at {_pct(barren)} is a caution because root shape quality depends on friable fertile soil.")
    if water >= 5:
        parts.append(f"Water presence at {_pct(water)} supports irrigation, but over-irrigation can crack carrot roots.")
    if 8 <= forest <= 25:
        parts.append(f"Forest coverage at {_pct(forest)} may reflect a cooler microclimate favorable for carrot quality.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for carrot (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 84 — Papaya
# ---------------------------------------------------------------------------

def explain_papaya(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; urban = obs[I_URBAN]; water = obs[I_WATER]; forest = obs[I_FOREST]
    parts = []
    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} supports papaya orchard planning.")
    if urban >= 8:
        parts.append(f"Urban share at {_pct(urban)} helps papaya through fast market access for perishable fruit.")
    if water >= 5:
        parts.append(f"Water presence at {_pct(water)} supports papaya growth, but root-zone drainage is essential.")
    if forest >= 12:
        parts.append(f"Forest coverage at {_pct(forest)} can indicate humid conditions favorable for papaya if disease is managed.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for papaya (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 85 — Guava
# ---------------------------------------------------------------------------

def explain_guava(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; barren = obs[I_BARREN]; urban = obs[I_URBAN]; forest = obs[I_FOREST]
    parts = []
    if agri >= 35:
        parts.append(f"Farmland at {_pct(agri)} provides a workable base for guava orchards.")
    if barren >= 15:
        parts.append(f"Barren share at {_pct(barren)} is generally manageable for guava with pit enrichment.")
    if urban >= 8:
        parts.append(f"Urban coverage at {_pct(urban)} is favorable for guava's direct fresh-fruit market flow.")
    if forest >= 12:
        parts.append(f"Forest at {_pct(forest)} can support humidity but may raise pest pressure if canopy ventilation is poor.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for guava (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 86 — Pomegranate / Anar
# ---------------------------------------------------------------------------

def explain_pomegranate(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; barren = obs[I_BARREN]; range_ = obs[I_RANGE]; water = obs[I_WATER]
    parts = []
    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} supports pomegranate orchard management.")
    if barren >= 20 or range_ >= 20:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) align with pomegranate's dryland adaptability.")
    if water >= 6:
        parts.append(f"Water share at {_pct(water)} supports irrigation access, but over-wetting can increase fruit cracking and disease.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for pomegranate (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 87 — Grapes
# ---------------------------------------------------------------------------

def explain_grapes(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; barren = obs[I_BARREN]; water = obs[I_WATER]; urban = obs[I_URBAN]
    parts = []
    if agri >= 45:
        parts.append(f"Farmland at {_pct(agri)} is a strong prerequisite for managed grape vineyards.")
    if barren >= 15:
        parts.append(f"Barren land at {_pct(barren)} can be workable for grapes on well-drained soils with fertigation.")
    if water >= 5:
        parts.append(f"Water presence at {_pct(water)} supports irrigation reliability, critical for vineyard scheduling.")
    if urban >= 8:
        parts.append(f"Urban share at {_pct(urban)} supports marketing and cold-chain movement for table grapes.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for grapes (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 88 — Apple
# ---------------------------------------------------------------------------

def explain_apple(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; forest = obs[I_FOREST]; water = obs[I_WATER]; urban = obs[I_URBAN]
    parts = []
    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} supports apple orchard structuring.")
    if forest >= 20:
        parts.append(f"Forest coverage at {_pct(forest)} may indicate cooler hill conditions compatible with apple systems.")
    if water >= 3:
        parts.append(f"Water presence at {_pct(water)} supports irrigation and frost management needs.")
    if urban >= 8:
        parts.append(f"Urban share at {_pct(urban)} can improve access to storage and sorting infrastructure.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for apple (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 90 — Lemon / Lime
# ---------------------------------------------------------------------------

def explain_lemon_lime(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; barren = obs[I_BARREN]; urban = obs[I_URBAN]; water = obs[I_WATER]
    parts = []
    if agri >= 35:
        parts.append(f"Farmland at {_pct(agri)} supports lemon/lime orchard layout and management.")
    if barren >= 15:
        parts.append(f"Barren share at {_pct(barren)} can be managed for citrus with pit enrichment and mulching.")
    if water >= 5:
        parts.append(f"Water presence at {_pct(water)} supports irrigation, while drainage must prevent root rot.")
    if urban >= 8:
        parts.append(f"Urban coverage at {_pct(urban)} is favorable for lemon/lime's continuous market demand.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for lemon/lime (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 91 — Sapota / Chiku
# ---------------------------------------------------------------------------

def explain_sapota(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; forest = obs[I_FOREST]; urban = obs[I_URBAN]; water = obs[I_WATER]
    parts = []
    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} provides a practical base for sapota orchards.")
    if 10 <= forest <= 30:
        parts.append(f"Forest share at {_pct(forest)} indicates a humid profile suitable for sapota growth.")
    if water >= 3:
        parts.append(f"Water presence at {_pct(water)} supports consistent fruiting in sapota.")
    if urban >= 8:
        parts.append(f"Urban coverage at {_pct(urban)} is useful for fresh-fruit sales and distribution.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for sapota/chiku (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 92 — Pineapple
# ---------------------------------------------------------------------------

def explain_pineapple(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; forest = obs[I_FOREST]; water = obs[I_WATER]; barren = obs[I_BARREN]
    parts = []
    if agri >= 35:
        parts.append(f"Farmland at {_pct(agri)} supports pineapple block establishment.")
    if forest >= 20:
        parts.append(f"Forest coverage at {_pct(forest)} can indicate humid conditions favorable for pineapple.")
    if water >= 3:
        parts.append(f"Water presence at {_pct(water)} supports moisture needs, though waterlogging should be avoided.")
    if barren >= 12:
        parts.append(f"Barren share at {_pct(barren)} signals the need for organic matter and mulch management.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for pineapple (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 93 — Avocado
# ---------------------------------------------------------------------------

def explain_avocado(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; forest = obs[I_FOREST]; water = obs[I_WATER]; urban = obs[I_URBAN]
    parts = []
    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} is a workable base for avocado orchards.")
    if forest >= 20:
        parts.append(f"Forest share at {_pct(forest)} can align with avocado's humid subtropical preference.")
    if water >= 3:
        parts.append(f"Water presence at {_pct(water)} supports irrigation planning for avocado establishment years.")
    if urban >= 8:
        parts.append(f"Urban coverage at {_pct(urban)} can support market linkage for premium avocado fruit.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for avocado (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 94 — Strawberry
# ---------------------------------------------------------------------------

def explain_strawberry(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; urban = obs[I_URBAN]; water = obs[I_WATER]; forest = obs[I_FOREST]
    parts = []
    if agri >= 40:
        parts.append(f"Farmland at {_pct(agri)} supports strawberry cultivation with raised beds and fertigation.")
    if urban >= 12:
        parts.append(f"Urban share at {_pct(urban)} is a strong commercial advantage for strawberry's short shelf life.")
    if water >= 3:
        parts.append(f"Water presence at {_pct(water)} supports irrigation, though canopy humidity must be controlled.")
    if forest >= 10:
        parts.append(f"Forest coverage at {_pct(forest)} may indicate cooler microclimate, beneficial for berry quality.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for strawberry (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 95 — Olive
# ---------------------------------------------------------------------------

def explain_olive(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; barren = obs[I_BARREN]; range_ = obs[I_RANGE]; water = obs[I_WATER]
    parts = []
    if agri >= 30:
        parts.append(f"Agricultural land at {_pct(agri)} supports olive orchard establishment.")
    if barren >= 25 or range_ >= 20:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) match olive's tolerance to dry conditions.")
    if water >= 4:
        parts.append(f"Water presence at {_pct(water)} supports early orchard growth, though olives tolerate moderate water stress later.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for olive (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 97 — Jackfruit / Kathal
# ---------------------------------------------------------------------------

def explain_jackfruit(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; forest = obs[I_FOREST]; water = obs[I_WATER]; urban = obs[I_URBAN]
    parts = []
    if agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} supports jackfruit orchard development.")
    if forest >= 20:
        parts.append(f"Forest share at {_pct(forest)} indicates humid tropical conditions favorable for jackfruit.")
    if water >= 3:
        parts.append(f"Water presence at {_pct(water)} supports stable growth, especially in establishment years.")
    if urban >= 8:
        parts.append(f"Urban coverage at {_pct(urban)} helps market access for both mature and tender jackfruit products.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for jackfruit (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 98 — Litchi
# ---------------------------------------------------------------------------

def explain_litchi(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; forest = obs[I_FOREST]; water = obs[I_WATER]; urban = obs[I_URBAN]
    parts = []
    if agri >= 35:
        parts.append(f"Agricultural land at {_pct(agri)} provides a workable base for litchi orchards.")
    if 15 <= forest <= 35:
        parts.append(f"Forest coverage at {_pct(forest)} supports humid conditions generally suitable for litchi.")
    if water >= 3:
        parts.append(f"Water presence at {_pct(water)} supports flowering and fruit development stability.")
    if urban >= 8:
        parts.append(f"Urban share at {_pct(urban)} can improve post-harvest movement for this highly perishable fruit.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for litchi (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 99 — Moringa / Drumstick
# ---------------------------------------------------------------------------

def explain_moringa(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    agri = obs[I_AGRI]; barren = obs[I_BARREN]; range_ = obs[I_RANGE]; urban = obs[I_URBAN]
    parts = []
    if agri >= 30:
        parts.append(f"Farmland at {_pct(agri)} supports moringa cultivation for pods and leaf markets.")
    if barren >= 20 or range_ >= 20:
        parts.append(f"Barren ({_pct(barren)}) and rangeland ({_pct(range_)}) align with moringa's dryland resilience.")
    if urban >= 8:
        parts.append(f"Urban coverage at {_pct(urban)} is useful for frequent harvesting and local market supply.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for moringa (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Crop 100 — Jatropha (Biofuel)
# ---------------------------------------------------------------------------

def explain_jatropha(obs: np.ndarray, fav: list, score: float, contribs: List[Dict]) -> str:
    barren = obs[I_BARREN]; range_ = obs[I_RANGE]; agri = obs[I_AGRI]; water = obs[I_WATER]
    parts = []
    if barren >= 30:
        parts.append(f"Barren land at {_pct(barren)} is a strong positive for jatropha's marginal-land biofuel profile.")
    if range_ >= 25:
        parts.append(f"Rangeland at {_pct(range_)} further supports jatropha suitability in low-input landscapes.")
    if agri >= 25:
        parts.append(f"Agricultural share at {_pct(agri)} helps management logistics, though jatropha is usually targeted at non-prime land.")
    if water >= 5:
        parts.append(f"Water presence at {_pct(water)} can support early establishment but high moisture is not required long term.")
    verdict = _confidence_phrase(score)
    parts.append(f"Overall, your land is {verdict} for jatropha (score {round(score)}/100).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Registry: maps crop_id → explanation builder function
# ---------------------------------------------------------------------------

EXPLANATION_BUILDERS: dict = {
    1:  explain_rice,
    2:  explain_wheat,
    3:  explain_maize,
    4:  explain_sorghum,
    5:  explain_pearl_millet,
    6:  explain_finger_millet,
    7:  explain_barley,
    8:  explain_oats,
    9:  explain_rye,
    10: explain_triticale,
    11: explain_teff,
    12: explain_foxtail_millet,
    13: explain_kodo_millet,
    14: explain_little_millet,
    15: explain_proso_millet,
    16: explain_chickpea,
    17: explain_pigeon_pea,
    18: explain_lentil,
    19: explain_green_gram,
    20: explain_black_gram,
    21: explain_kidney_bean,
    22: explain_cowpea,
    23: explain_field_pea,
    24: explain_soybean,
    25: explain_moth_bean,
    26: explain_horse_gram,
    27: explain_cluster_bean,
    28: explain_groundnut,
    29: explain_mustard,
    30: explain_sunflower,
    31: explain_sesame,
    32: explain_linseed,
    33: explain_castor,
    34: explain_safflower,
    35: explain_niger,
    36: explain_palm_oil,
    37: explain_coconut,
    38: explain_cotton,
    39: explain_cotton_desi,
    40: explain_jute,
    41: explain_hemp,
    42: explain_kenaf,
    43: explain_sugarcane,
    44: explain_sugar_beet,
    45: explain_tea,
    46: explain_coffee_arabica,
    47: explain_coffee_robusta,
    48: explain_rubber,
    49: explain_tobacco,
    50: explain_cocoa,
    51: explain_cashew,
    52: explain_black_pepper,
    53: explain_cardamom_large,
    54: explain_cardamom_small,
    55: explain_turmeric,
    56: explain_ginger,
    57: explain_chili,
    58: explain_coriander,
    59: explain_cumin,
    60: explain_fenugreek,
    61: explain_clove,
    62: explain_nutmeg,
    63: explain_vanilla,
    64: explain_saffron,
    65: explain_fennel,
    66: explain_potato,
    67: explain_onion,
    68: explain_tomato,
    69: explain_brinjal,
    70: explain_cabbage,
    71: explain_cauliflower,
    72: explain_garlic,
    73: explain_okra,
    74: explain_pumpkin,
    75: explain_cucumber,
    76: explain_cassava,
    77: explain_sweet_potato,
    78: explain_yam,
    79: explain_taro,
    80: explain_spinach,
    81: explain_carrot,
    82: explain_mango,
    83: explain_banana,
    84: explain_papaya,
    85: explain_guava,
    86: explain_pomegranate,
    87: explain_grapes,
    88: explain_apple,
    89: explain_orange,
    90: explain_lemon_lime,
    91: explain_sapota,
    92: explain_pineapple,
    93: explain_avocado,
    94: explain_strawberry,
    95: explain_olive,
    96: explain_date_palm,
    97: explain_jackfruit,
    98: explain_litchi,
    99: explain_moringa,
    100: explain_jatropha,
}


def build_explanation(crop_id: int, obs: np.ndarray, fav: list,
                      score: float, contribs: List[Dict]) -> str | None:
    builder = EXPLANATION_BUILDERS.get(crop_id)
    if builder is None:
        return None
    return builder(obs, fav, score, contribs)


def _confidence_band(score: float) -> str:
    if score >= 80:
        return "High"
    if score >= 60:
        return "Moderate"
    return "Low"


def generate_explanation(result: Dict) -> Dict:
    """
    Backward-compatible explanation pack for recommend_crops(...) output.
    """
    if not isinstance(result, dict):
        return {
            "summary": "No explanation available.",
            "land_analysis": "Invalid recommendation payload.",
            "indices_explanation": "Indices unavailable.",
            "regime_explanation": "Water regime unavailable.",
            "soil_explanation": "Soil class unavailable.",
            "market_explanation": "Market class unavailable.",
            "crop_explanations": [],
            "recommendations_summary": "No crop recommendations were generated.",
        }

    status = result.get("status", "ok")
    ranked = result.get("ranked_crops", []) or []

    if status == "halted":
        return {
            "summary": "Recommendation halted due to land profile constraints.",
            "land_analysis": result.get("halt_message", "Crop recommendation was halted."),
            "indices_explanation": f"Computed indices: {result.get('indices', {})}",
            "regime_explanation": f"Detected water regime: {result.get('water_regime', 'Unknown')}",
            "soil_explanation": f"Detected soil class: {result.get('soil_class', 'Unknown')}",
            "market_explanation": f"Detected market access class: {result.get('market_class', 'Unknown')}",
            "crop_explanations": [],
            "recommendations_summary": "No ranked crops available because the workflow was halted.",
        }

    crop_explanations = []
    for item in ranked:
        score = float(item.get("score", 0.0) or 0.0)
        reasoning = str(item.get("reasoning", "")).strip()
        if not reasoning:
            reasoning = (
                f"{item.get('crop', 'This crop')} is ranked based on compatibility with the "
                f"current land-cover profile and terrain fit."
            )

        pros = []
        if score >= 70:
            pros.append("Strong overall suitability")
        if str(item.get("regime_match", "")).lower() == "strong":
            pros.append("Good water-regime alignment")
        if item.get("risk_tier") == "Best Fit":
            pros.append("Top-tier recommendation")

        cons = []
        if item.get("marginal"):
            cons.append("Marginal suitability; needs careful management")
        if str(item.get("regime_match", "")).lower() == "weak":
            cons.append("Weak water-regime match")
        if score < 50:
            cons.append("Lower expected resilience/yield")

        crop_explanations.append({
            "rank": item.get("rank"),
            "crop": item.get("crop"),
            "score": score,
            "confidence": _confidence_band(score),
            "reasoning": reasoning,
            "pros": pros,
            "cons": cons,
        })

    top_text = ", ".join(c.get("crop", "Unknown") for c in ranked[:3]) or "None"
    return {
        "summary": f"Generated {len(ranked)} crop recommendations.",
        "land_analysis": (
            f"Terrain: {result.get('terrain_classification', {}).get('name', 'Mixed Terrain')}. "
            f"Flags: {', '.join(result.get('flags', [])) if result.get('flags') else 'No critical warnings.'}"
        ),
        "indices_explanation": f"Indices snapshot: {result.get('indices', {})}",
        "regime_explanation": f"Water regime identified as {result.get('water_regime', 'Unknown')}.",
        "soil_explanation": f"Soil class identified as {result.get('soil_class', 'Unknown')}.",
        "market_explanation": f"Market access class identified as {result.get('market_class', 'Unknown')}.",
        "crop_explanations": crop_explanations,
        "recommendations_summary": f"Top recommendations: {top_text}.",
    }
