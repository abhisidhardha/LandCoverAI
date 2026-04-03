"""
crop_config.py
==============
Configuration parameters for the crop recommendation engine.
Modify these values to customize the recommendation behavior.
"""

# ==============================================================================
# FEASIBILITY GATE THRESHOLDS
# ==============================================================================

# Urban dominated threshold (%)
URBAN_HALT_THRESHOLD = 65

# Barren wasteland thresholds (%)
BARREN_HALT_THRESHOLD = 70
BARREN_HALT_MIN_WATER = 5
BARREN_HALT_MIN_AGRI = 5

# Water inundated threshold (%)
WATER_HALT_THRESHOLD = 60

# Minimum cultivation feasibility index
MIN_CFI_THRESHOLD = 15
MARGINAL_CFI_THRESHOLD = 30

# ==============================================================================
# WATER REGIME THRESHOLDS
# ==============================================================================

# Water-rich regime
WATER_RICH_MIN_MAI = 65
WATER_RICH_MIN_WATER = 20
FLOOD_RISK_WATER_THRESHOLD = 40

# Humid regime
HUMID_MIN_MAI = 45
HUMID_MAX_MAI = 65
HUMID_MIN_FOREST = 15

# Sub-humid regime
SUB_HUMID_MIN_MAI = 30
SUB_HUMID_MAX_MAI = 45

# Semi-arid regime
SEMI_ARID_MIN_MAI = 15
SEMI_ARID_MAX_MAI = 30
SEMI_ARID_MIN_ASI = 30

# Arid regime
ARID_MAX_MAI = 15
ARID_MIN_ASI = 50

# ==============================================================================
# SOIL CLASS THRESHOLDS
# ==============================================================================

# Productive soil
PRODUCTIVE_MIN_SHI = 60

# Moderate soil
MODERATE_MIN_SHI = 35
MODERATE_MAX_SHI = 60

# Degraded soil
DEGRADED_MAX_SHI = 35
DEGRADED_MIN_BARREN = 20

# Reclamation zone
RECLAMATION_MAX_SHI = 20
RECLAMATION_MIN_BARREN = 45

# Agroforestry overlay
AGROFORESTRY_MIN_FOREST = 25
AGROFORESTRY_MIN_AGRI = 15

# ==============================================================================
# MARKET CLASS THRESHOLDS
# ==============================================================================

# High-value market
HIGH_VALUE_MIN_MEI = 55

# Commercial market
COMMERCIAL_MIN_MEI = 25
COMMERCIAL_MAX_MEI = 55

# Peri-urban zone
PERI_URBAN_MIN_URBAN = 20
PERI_URBAN_MIN_AGRI = 20

# Export potential
EXPORT_MIN_AGRI = 40
EXPORT_MIN_MEI = 40
EXPORT_MIN_SHI = 50

# ==============================================================================
# SCORING WEIGHTS
# ==============================================================================

# Bonus points
MAI_MATCH_BONUS_MAX = 20
SHI_MATCH_BONUS_MAX = 15
CFI_BONUS_MAX = 10
MEI_MARKET_BONUS_MAX = 20
AFI_AGROFORESTRY_BONUS_MAX = 10
REGIME_MATCH_BONUS = 10
PRODUCTIVE_SOIL_BOOST = 15

# Penalty points
ASI_STRESS_PENALTY_MAX = 20
SOIL_DEGRADATION_PENALTY = 20
MARKET_MISMATCH_PENALTY = 10
RECLAMATION_NON_BUILDER_PENALTY = 30
DEGRADED_VEGETABLE_PENALTY = 20

# ==============================================================================
# CONFIDENCE THRESHOLDS
# ==============================================================================

# Score ranges for confidence levels
HIGH_CONFIDENCE_MIN = 75
MODERATE_CONFIDENCE_MIN = 50
LOW_CONFIDENCE_MIN = 30
# Scores below LOW_CONFIDENCE_MIN are removed from output

# ==============================================================================
# COMPOSITE INDEX WEIGHTS
# ==============================================================================

# MAI (Moisture Availability Index) weights
MAI_WATER_WEIGHT = 1.0
MAI_FOREST_WEIGHT = 0.6
MAI_AGRI_WEIGHT = 0.3
MAI_RANGE_WEIGHT = 0.1

# SHI (Soil Health Index) weights
SHI_AGRI_WEIGHT = 1.0
SHI_FOREST_WEIGHT = 0.7
SHI_BARREN_WEIGHT = -1.2
SHI_URBAN_WEIGHT = -0.4

# ASI (Aridity-Stress Index) weights
ASI_BARREN_WEIGHT = 1.0
ASI_RANGE_WEIGHT = 0.6
ASI_WATER_WEIGHT = -0.5
ASI_FOREST_WEIGHT = -0.3

# CFI (Cultivation Feasibility Index) weights
CFI_AGRI_WEIGHT = 1.0
CFI_RANGE_WEIGHT = 0.5
CFI_BARREN_WEIGHT = 0.2
CFI_URBAN_WEIGHT = -0.5
CFI_FOREST_WEIGHT = -0.2

# MEI (Market & Economic Index) weights
MEI_URBAN_WEIGHT = 1.0
MEI_AGRI_WEIGHT = 0.3

# AFI (Agroforestry Potential Index) weights
AFI_FOREST_WEIGHT = 1.0
AFI_WATER_WEIGHT = 0.4
AFI_RANGE_WEIGHT = 0.2
AFI_BARREN_WEIGHT = -0.5

# ==============================================================================
# SUITABILITY FILTER PARAMETERS
# ==============================================================================

# Allow crops with CFI up to 50% below minimum requirement
CFI_SOFT_GATE_FACTOR = 0.5

# ==============================================================================
# OUTPUT PARAMETERS
# ==============================================================================

# Default number of crops to return
DEFAULT_TOP_N = 10

# Maximum number of crops to return
MAX_TOP_N = 50

# ==============================================================================
# USAGE NOTES
# ==============================================================================
"""
To customize the recommendation engine:

1. Modify threshold values above
2. Restart the Flask server
3. Test with test_recommender.py

Example customizations:

- Make the system more conservative:
  * Increase HIGH_CONFIDENCE_MIN to 80
  * Increase MODERATE_CONFIDENCE_MIN to 60
  * Increase feasibility gate thresholds

- Make the system more permissive:
  * Decrease confidence thresholds
  * Decrease feasibility gate thresholds
  * Increase CFI_SOFT_GATE_FACTOR to 0.7

- Adjust for specific regions:
  * Modify water regime thresholds for local climate
  * Adjust soil class thresholds based on local soil data
  * Customize market class thresholds for local economics

- Fine-tune scoring:
  * Adjust bonus/penalty weights
  * Modify composite index weights
  * Change regime match bonus
"""
