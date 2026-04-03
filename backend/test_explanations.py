"""
Test crop explanations system
"""

from crop_recommender import recommend_crops
from crop_explanations import generate_explanation
import json

# Test with balanced agricultural land
print("=" * 80)
print("Testing Crop Explanations System")
print("=" * 80)

result = recommend_crops({
    "urban_land": 5,
    "agriculture": 45,
    "rangeland": 15,
    "forest": 20,
    "water": 10,
    "barren": 5,
}, top_n=5)

explanations = generate_explanation(result)

print("\nSUMMARY:")
print(explanations["summary"])

print("\n" + "=" * 80)
print(explanations["land_analysis"])

print("\n" + "=" * 80)
print(explanations["indices_explanation"])

print("\n" + "=" * 80)
print(explanations["regime_explanation"])

print("\n" + "=" * 80)
print(explanations["soil_explanation"])

print("\n" + "=" * 80)
print(explanations["market_explanation"])

print("\n" + "=" * 80)
print(" TOP CROP RECOMMENDATIONS WITH REASONING:")
print("=" * 80)

for crop_exp in explanations["crop_explanations"]:
    print(f"\n{crop_exp['rank']}. {crop_exp['crop']} - Score: {crop_exp['score']:.1f} ({crop_exp['confidence']})")
    print(f"   {crop_exp['reasoning']}")
    print(f"   Pros: {', '.join(crop_exp['pros'])}")
    if crop_exp['cons']:
        print(f"    Cons: {', '.join(crop_exp['cons'])}")

print("\n" + "=" * 80)
print(explanations["recommendations_summary"])

print("\n" + "=" * 80)
print("Explanation system test complete!")

