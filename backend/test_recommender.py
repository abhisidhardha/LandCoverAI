"""
Test script for crop_recommender.py
"""

from crop_recommender import recommend_crops
import json

# Test Case 1: Balanced agricultural land
print("=" * 80)
print("TEST 1: Balanced Agricultural Land")
print("=" * 80)
result1 = recommend_crops({
    "urban_land": 5,
    "agriculture": 45,
    "rangeland": 15,
    "forest": 20,
    "water": 10,
    "barren": 5,
})
print(json.dumps(result1, indent=2))

# Test Case 2: Arid/Semi-arid region
print("\n" + "=" * 80)
print("TEST 2: Arid/Semi-Arid Region")
print("=" * 80)
result2 = recommend_crops({
    "urban_land": 2,
    "agriculture": 15,
    "rangeland": 30,
    "forest": 5,
    "water": 3,
    "barren": 45,
})
print(json.dumps(result2, indent=2))

# Test Case 3: Water-rich region
print("\n" + "=" * 80)
print("TEST 3: Water-Rich Region")
print("=" * 80)
result3 = recommend_crops({
    "urban_land": 3,
    "agriculture": 25,
    "rangeland": 5,
    "forest": 30,
    "water": 35,
    "barren": 2,
})
print(json.dumps(result3, indent=2))

# Test Case 4: Urban dominated (should halt)
print("\n" + "=" * 80)
print("TEST 4: Urban Dominated (Halt Expected)")
print("=" * 80)
result4 = recommend_crops({
    "urban_land": 75,
    "agriculture": 10,
    "rangeland": 5,
    "forest": 5,
    "water": 3,
    "barren": 2,
})
print(json.dumps(result4, indent=2))

# Test Case 5: Degraded soil
print("\n" + "=" * 80)
print("TEST 5: Degraded Soil")
print("=" * 80)
result5 = recommend_crops({
    "urban_land": 5,
    "agriculture": 20,
    "rangeland": 15,
    "forest": 10,
    "water": 5,
    "barren": 45,
})
print(json.dumps(result5, indent=2))
