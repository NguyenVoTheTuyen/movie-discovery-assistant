"""
baseline_eval.py — Bayesian m Sensitivity Analysis & Baseline Evaluation.

Covers:
    1. Sensitivity analysis for Bayesian m ∈ {5, 10, 30}:
       - Catalog coverage (eligible movies)
       - Top-10 overlap with Hybrid recommendations
       - Genre diversity in top-10

    2. Popularity baseline vs Hybrid for 3 evaluation users.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bayesian_engine import sensitivity_analysis, get_top_bayesian
from data_loader import (
    build_rating_stats,
    get_user_watched_ids,
    get_user_rating_count,
)
from hybrid_engine import get_recommendations


EVAL_USERS = [1, 15, 30]


def run_m_sensitivity() -> None:
    """Compare m ∈ {5, 10, 30} for Bayesian Weighted Rating."""
    print("=" * 60)
    print("Bayesian m Sensitivity Analysis")
    print("=" * 60)

    results = sensitivity_analysis()

    for m, info in results.items():
        print(f"\nm = {m}:")
        print(f"  Eligible movies : {info['eligible_count']:,} / 5,126 ({info['eligible_pct']}%)")
        print(f"  Top-10 titles   : {info['top10_titles'][:5]}")

    print("\n\nDecision: m=10 chosen")
    print("  m=30 → only 13.7% of catalog eligible — too restrictive")
    print("  m=10 → 32.4% eligible — good balance quality vs. coverage")
    print("  m=5  → 47.2% eligible — too permissive, includes low-vote movies")


def run_top10_overlap(user_id: int) -> dict:
    """Compare Hybrid top-10 vs Bayesian top-10 overlap."""
    watched = get_user_watched_ids(user_id)

    hybrid_recs, _ = get_recommendations(user_id, n=10)
    hybrid_ids = {r["movieId"] for r in hybrid_recs}

    bayesian_recs = get_top_bayesian(n=10, exclude_ids=watched)
    bayesian_ids = {r["movieId"] for r in bayesian_recs}

    overlap = hybrid_ids & bayesian_ids
    overlap_pct = len(overlap) / 10 * 100

    return {
        "user_id": user_id,
        "hybrid_count": len(hybrid_ids),
        "bayesian_count": len(bayesian_ids),
        "overlap": len(overlap),
        "overlap_pct": round(overlap_pct, 1),
        "overlap_titles": [r["title"] for r in hybrid_recs if r["movieId"] in overlap],
    }


def compute_genre_diversity(recs: list[dict]) -> dict:
    """Compute genre diversity stats for a list of recommendations."""
    genre_counts: dict[str, int] = {}
    for m in recs:
        for g in m.get("genres", "").split("|"):
            if g and g != "(no genres listed)":
                genre_counts[g] = genre_counts.get(g, 0) + 1
    return {
        "unique_genres": len(genre_counts),
        "genre_distribution": sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:8],
        "max_single_genre": max(genre_counts.values()) if genre_counts else 0,
    }


def main():
    run_m_sensitivity()

    print("\n\n" + "=" * 60)
    print("Top-10 Overlap: Hybrid vs. Popularity Baseline")
    print("=" * 60)

    for uid in EVAL_USERS:
        overlap_info = run_top10_overlap(uid)
        print(f"\nUser {uid}: {overlap_info['overlap']}/10 movies overlap ({overlap_info['overlap_pct']}%)")
        if overlap_info["overlap_titles"]:
            print(f"  Shared: {overlap_info['overlap_titles']}")

    print("\n\n" + "=" * 60)
    print("Genre Diversity: Hybrid vs. Popularity Baseline")
    print("=" * 60)

    for uid in EVAL_USERS:
        watched = get_user_watched_ids(uid)
        hybrid_recs, _ = get_recommendations(uid, n=10)
        bayesian_recs = get_top_bayesian(n=10, exclude_ids=watched)

        hybrid_div = compute_genre_diversity(hybrid_recs)
        bayesian_div = compute_genre_diversity(bayesian_recs)

        rating_counts = get_user_rating_count()
        count = rating_counts.get(uid, 0)
        print(f"\nUser {uid} ({count} ratings):")
        print(f"  Hybrid     → {hybrid_div['unique_genres']} unique genres, max single: {hybrid_div['max_single_genre']}")
        print(f"  Popularity → {bayesian_div['unique_genres']} unique genres, max single: {bayesian_div['max_single_genre']}")


if __name__ == "__main__":
    main()
