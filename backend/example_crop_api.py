"""
Example: Using the Crop Recommendation API
===========================================

This script demonstrates how to use the crop recommendation endpoint
after getting land-cover segmentation results.
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:5000"
TOKEN = "your_auth_token_here"  # Get this from /api/login

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Example 1: Get crop recommendations from land-cover percentages
print("=" * 80)
print("Example 1: Direct Crop Recommendation")
print("=" * 80)

landcover_data = {
    "percentages": {
        "urban_land": 5,
        "agriculture": 45,
        "rangeland": 15,
        "forest": 20,
        "water": 10,
        "barren": 5,
        "unknown": 0
    },
    "top_n": 10
}

response = requests.post(
    f"{BASE_URL}/api/recommend-crops",
    headers=headers,
    json=landcover_data
)

if response.status_code == 200:
    result = response.json()
    print(f"\nStatus: {result['status']}")
    print(f"Water Regime: {result['water_regime']}")
    print(f"Soil Class: {result['soil_class']}")
    print(f"Market Class: {result['market_class']}")
    print(f"\nIndices:")
    for key, value in result['indices'].items():
        print(f"  {key}: {value}")
    
    print(f"\nFlags:")
    for flag in result['flags']:
        print(f"  - {flag}")
    
    print(f"\nTop {len(result['ranked_crops'])} Recommended Crops:")
    for crop in result['ranked_crops']:
        confidence = "HIGH" if crop['score'] >= 75 else "MODERATE" if crop['score'] >= 50 else "LOW"
        print(f"  {crop['rank']}. {crop['crop']} ({crop['category']}, {crop['season']})")
        print(f"     Score: {crop['score']} {confidence}")
        print(f"     Regime Match: {crop['regime_match']}, Marginal: {crop['marginal']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)


# Example 2: Full workflow - Predict from coordinates + Get crop recommendations
print("\n" + "=" * 80)
print("Example 2: Full Workflow (Coordinates → Segmentation → Crop Recommendation)")
print("=" * 80)

# Step 1: Get segmentation from coordinates
coords_data = {
    "lat": 28.6139,  # New Delhi
    "lon": 77.2090,
    "radius_m": 500,
    "size": 512
}

print("\nStep 1: Fetching satellite image and running segmentation...")
seg_response = requests.post(
    f"{BASE_URL}/api/predict/coordinates",
    headers=headers,
    json=coords_data
)

if seg_response.status_code == 200:
    seg_result = seg_response.json()
    print("✓ Segmentation complete")
    
    # Extract land-cover percentages
    landcover_pct = seg_result.get('landcover_percentages', {})
    print(f"\nLand-Cover Distribution:")
    for cls, pct in landcover_pct.items():
        if pct > 0:
            print(f"  {cls}: {pct}%")
    
    # Step 2: Get crop recommendations
    print("\nStep 2: Getting crop recommendations...")
    crop_data = {
        "percentages": landcover_pct,
        "top_n": 5
    }
    
    crop_response = requests.post(
        f"{BASE_URL}/api/recommend-crops",
        headers=headers,
        json=crop_data
    )
    
    if crop_response.status_code == 200:
        crop_result = crop_response.json()
        print("✓ Crop recommendation complete")
        
        if crop_result['status'] == 'halted':
            print(f"\nRecommendation Halted: {crop_result['halt_message']}")
        else:
            print(f"\nAnalysis Results:")
            print(f"   Water Regime: {crop_result['water_regime']}")
            print(f"   Soil Class: {crop_result['soil_class']}")
            print(f"   Market Class: {crop_result['market_class']}")
            
            if crop_result['flags']:
                print(f"\nWarnings:")
                for flag in crop_result['flags']:
                    print(f"   - {flag}")
            
            print(f"\n Top {len(crop_result['ranked_crops'])} Recommended Crops:")
            for crop in crop_result['ranked_crops']:
                marker = "HIGH" if crop['score'] >= 75 else "MODERATE" if crop['score'] >= 50 else "LOW"
                print(f"   [{marker}] {crop['rank']}. {crop['crop']}")
                print(f"      Category: {crop['category']} | Season: {crop['season']}")
                print(f"      Score: {crop['score']:.1f}/100")
    else:
        print(f"Error in crop recommendation: {crop_response.status_code}")
        print(crop_response.text)
else:
    print(f"Error in segmentation: {seg_response.status_code}")
    print(seg_response.text)


# Example 3: Arid region (should recommend drought-tolerant crops)
print("\n" + "=" * 80)
print("Example 3: Arid Region Analysis")
print("=" * 80)

arid_data = {
    "percentages": {
        "urban_land": 2,
        "agriculture": 10,
        "rangeland": 35,
        "forest": 3,
        "water": 2,
        "barren": 48,
        "unknown": 0
    },
    "top_n": 5
}

response = requests.post(
    f"{BASE_URL}/api/recommend-crops",
    headers=headers,
    json=arid_data
)

if response.status_code == 200:
    result = response.json()
    print(f"\nWater Regime: {result['water_regime']}")
    print(f"Aridity-Stress Index (ASI): {result['indices']['ASI']}")
    print(f"Moisture Availability Index (MAI): {result['indices']['MAI']}")
    
    print(f"\nRecommended Drought-Tolerant Crops:")
    for crop in result['ranked_crops']:
        print(f"  {crop['rank']}. {crop['crop']} (Score: {crop['score']})")
else:
    print(f"Error: {response.status_code}")
    print(response.text)

