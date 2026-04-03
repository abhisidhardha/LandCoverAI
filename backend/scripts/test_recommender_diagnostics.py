import itertools
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

# Allow running this file directly from backend/scripts
THIS_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from crop_recommender import CROP_DATA, _compute_suitability, get_recommender, recommend_crops


def _normalize_profile(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr.astype(float), 0.0, 100.0)
    total = float(np.sum(arr))
    if total <= 0:
        return arr
    return arr / total * 100.0


def _to_api_percentages(obs: np.ndarray) -> Dict[str, float]:
    # obs order: [urban, agri, barren, forest, rangeland, water]
    return {
        "urban_land": round(float(obs[0]), 2),
        "agriculture": round(float(obs[1]), 2),
        "barren": round(float(obs[2]), 2),
        "forest": round(float(obs[3]), 2),
        "rangeland": round(float(obs[4]), 2),
        "water": round(float(obs[5]), 2),
    }


def _profile_distance_l1(rec_a: Dict, rec_b: Dict) -> float:
    fa = rec_a["favorable"]
    fb = rec_b["favorable"]
    va = np.array([
        float(fa["urban"]),
        float(fa["agriculture"]),
        float(fa["barren"]),
        float(fa["forest"]),
        float(fa["rangeland"]),
        float(fa["water"]),
    ], dtype=float)
    vb = np.array([
        float(fb["urban"]),
        float(fb["agriculture"]),
        float(fb["barren"]),
        float(fb["forest"]),
        float(fb["rangeland"]),
        float(fb["water"]),
    ], dtype=float)
    return float(np.sum(np.abs(va - vb)))


def _check_rank_integrity(ranked: List[Dict]) -> List[str]:
    issues: List[str] = []
    if not ranked:
        return issues

    scores = [float(r["score"]) for r in ranked]
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1]:
            issues.append(f"Ranking not sorted at positions {i+1}->{i+2}: {scores[i]} < {scores[i+1]}")
            break

    crop_ids = [int(r["crop_id"]) for r in ranked]
    if len(crop_ids) != len(set(crop_ids)):
        issues.append("Duplicate crop IDs found in ranked output")

    # v5 uses soft diversity, so only flag extreme category concentration.
    cats = Counter(r["category"] for r in ranked)
    over_cap = {k: v for k, v in cats.items() if v > 6}
    if over_cap:
        issues.append(f"Category concentration too high (>6): {over_cap}")

    return issues


def _check_profile_similarity(recs: List[Dict], threshold: float = 12.0) -> List[str]:
    # Soft check: too many near-duplicate favorable profiles means diversity is weak.
    near_pairs: List[Tuple[int, int, float]] = []
    for i, j in itertools.combinations(range(len(recs)), 2):
        dist = _profile_distance_l1(recs[i], recs[j])
        if dist < threshold:
            near_pairs.append((i, j, round(dist, 2)))

    if len(near_pairs) >= 3:
        return [
            "High profile redundancy detected in selected recommendations: "
            f"{len(near_pairs)} pairs under L1 distance {threshold}."
        ]
    return []


def _water_sensitivity_checks() -> List[str]:
    issues: List[str] = []
    rice = next(c for c in CROP_DATA if c[0] == 1)
    millet = next(c for c in CROP_DATA if c[0] == 5)

    # Keep profile shape similar and vary water; agriculture absorbs the remainder.
    base = np.array([5.0, 0.0, 10.0, 5.0, 10.0, 0.0], dtype=float)
    water_levels = [0.0, 5.0, 10.0, 15.0, 20.0]

    rice_scores = []
    millet_scores = []
    for w in water_levels:
        obs = base.copy()
        obs[5] = w
        obs[1] = max(0.0, 100.0 - (obs[0] + obs[2] + obs[3] + obs[4] + obs[5]))
        obs = _normalize_profile(obs)
        rice_scores.append(_compute_suitability(obs, rice)[0])
        millet_scores.append(_compute_suitability(obs, millet)[0])

    # Rice should not drop when moving from very dry to moderately wet.
    if rice_scores[0] > rice_scores[2]:
        issues.append(f"Rice water sensitivity anomaly: score@0 water {rice_scores[0]} > score@10 water {rice_scores[2]}")

    # Arid millet should usually decline when water gets high.
    if millet_scores[-1] > millet_scores[1]:
        issues.append(
            f"Millet wetness anomaly: score@20 water {millet_scores[-1]} > score@5 water {millet_scores[1]}"
        )

    print("\nWater sensitivity trace")
    print("  water levels:", water_levels)
    print("  rice scores :", rice_scores)
    print("  millet scores:", millet_scores)

    return issues


def run_diagnostics() -> int:
    issues: List[str] = []
    warnings: List[str] = []

    scenarios = {
        "agriculture_dominant": np.array([3, 82, 6, 3, 4, 2], dtype=float),
        "water_dominant": np.array([2, 18, 5, 8, 5, 62], dtype=float),
        "barren_dominant": np.array([2, 28, 52, 4, 12, 2], dtype=float),
        "forest_dominant": np.array([2, 24, 5, 55, 10, 4], dtype=float),
        "rangeland_dominant": np.array([2, 30, 15, 8, 42, 3], dtype=float),
        "balanced_mixed": np.array([6, 34, 16, 15, 16, 13], dtype=float),
        "urban_hard_stop": np.array([85, 8, 2, 1, 2, 2], dtype=float),
        "agri_low_water": np.array([4, 72, 10, 4, 8, 2], dtype=float),
        "agri_medium_water": np.array([4, 62, 8, 4, 8, 14], dtype=float),
    }

    recommender = get_recommender()

    print("=== Crop Recommender Diagnostics ===")
    for name, obs in scenarios.items():
        obs = _normalize_profile(obs)
        api = _to_api_percentages(obs)
        result = recommend_crops(api, top_n=10)

        print(f"\nScenario: {name}")
        print("  Input:", api)
        print("  Status:", result.get("status"))

        if name == "urban_hard_stop":
            if result.get("status") != "halted":
                issues.append("urban_hard_stop scenario did not halt as expected")
            continue

        if result.get("status") != "ok":
            issues.append(f"Scenario {name} returned non-ok status: {result.get('status')}")
            continue

        ranked = result.get("ranked_crops", [])
        if not ranked:
            issues.append(f"Scenario {name} returned empty ranked_crops")
            continue

        print("  Top-3 nearest predictions:")
        for row in ranked[:3]:
            print(f"    - {row['crop']} ({row['category']}): score={row['score']} risk={row.get('prediction_risk')}")

        issues.extend([f"[{name}] {m}" for m in _check_rank_integrity(ranked)])

        # Deep similarity check uses recs from recommend_and_explain (contains favorable vectors).
        recs, _, _ = recommender.recommend_and_explain(obs, top_k=10)
        warnings.extend([f"[{name}] {m}" for m in _check_profile_similarity(recs, threshold=12.0)])

        # CI sanity checks
        for rec in ranked:
            ci = rec.get("confidence_interval")
            score = float(rec.get("score", 0.0))
            if ci and isinstance(ci, (list, tuple)) and len(ci) == 2:
                lo, hi = float(ci[0]), float(ci[1])
                if lo > hi:
                    issues.append(f"[{name}] CI invalid for {rec['crop']}: low>{'high'}")
                if score < lo - 15 or score > hi + 15:
                    warnings.append(
                        f"[{name}] score {score} far from CI [{lo}, {hi}] for {rec['crop']}"
                    )

    issues.extend(_water_sensitivity_checks())

    print("\n=== Diagnostic Summary ===")
    print(f"Issues   : {len(issues)}")
    print(f"Warnings : {len(warnings)}")

    if issues:
        print("\n[ISSUES]")
        for item in issues:
            print(" -", item)

    if warnings:
        print("\n[WARNINGS]")
        for item in warnings[:20]:
            print(" -", item)
        if len(warnings) > 20:
            print(f" - ... and {len(warnings) - 20} more warnings")

    # Persist report for easy review.
    report = {
        "issues": issues,
        "warnings": warnings,
    }
    out_path = os.path.abspath(os.path.join(BACKEND_DIR, "..", "eval_output", "recommender_diagnostics.json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report: {out_path}")

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostics())
