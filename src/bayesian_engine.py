"""
bayesian_engine.py — Bayesian Weighted Rating engine for cold-start users.

Formula:
    W = (v / (v + m)) × R + (m / (v + m)) × C

Where:
    v = vote count for the movie
    R = mean rating for the movie
    m = 10 (chosen after sensitivity analysis: m=30 → only 14% catalog eligible,
             m=10 → 32% eligible, m=5 → 47% eligible — good balance)
    C = 3.4947 (global mean, verified from actual data)

Used for:
    - Cold-start users (0 ratings)  → return top-20 by Bayesian score
    - Ranking quality signal in all other paths
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from data_loader import (
    GLOBAL_MEAN_RATING,
    build_movie_lookup,
    build_rating_stats,
    get_user_watched_ids,
    load_movies,
)

# Hyperparameters (see sensitivity analysis in evaluation/baseline_eval.py)
BAYESIAN_M: int = 10
GLOBAL_MEAN_C: float = GLOBAL_MEAN_RATING


def bayesian_score(vote_count: int, avg_rating: float, m: int = BAYESIAN_M, C: float = GLOBAL_MEAN_C) -> float:
    """Compute Bayesian Weighted Rating for a single movie."""
    return (vote_count / (vote_count + m)) * avg_rating + (m / (vote_count + m)) * C


def get_top_bayesian(
    n: int = 20,
    exclude_ids: Optional[set[int]] = None,
    min_votes: int = 10,
    genre_filter: Optional[list[str]] = None,
) -> list[dict]:
    """Return top-N movies ranked by Bayesian Weighted Rating.

    Args:
        n: Number of movies to return.
        exclude_ids: Set of movieIds to exclude (already-watched movies).
        min_votes: Minimum rating count to be eligible.
        genre_filter: If provided, only include movies containing at least one of these genres.

    Returns:
        List of dicts with movieId, title, genres, plot, tags, avg_rating,
        vote_count, bayesian_score.
    """
    exclude_ids = exclude_ids or set()
    stats = build_rating_stats()
    movies_lookup = build_movie_lookup()
    movies_df = load_movies()

    # Filter by minimum votes
    eligible = stats[
        (stats["vote_count"] >= min_votes) &
        (~stats["movieId"].isin(exclude_ids))
    ].copy()

    # Genre filter (if requested)
    if genre_filter:
        genre_set = set(g.lower() for g in genre_filter)
        def has_genre(movie_id: int) -> bool:
            movie = movies_lookup.get(movie_id, {})
            genres = movie.get("genres", "")
            if not genres or genres == "(no genres listed)":
                return False
            return any(g.lower() in genre_set for g in genres.split("|"))
        eligible = eligible[eligible["movieId"].apply(has_genre)]

    # Sort by Bayesian score descending
    eligible = eligible.sort_values("bayesian_score", ascending=False).head(n)

    results = []
    for _, row in eligible.iterrows():
        mid = int(row["movieId"])
        movie = movies_lookup.get(mid, {})
        results.append({
            "movieId": mid,
            "title": movie.get("title", f"Movie {mid}"),
            "year": movie.get("year", ""),
            "genres": movie.get("genres", ""),
            "plot": movie.get("plot", ""),
            "tags": movie.get("tags", []),
            "avg_rating": round(float(row["avg_rating"]), 2),
            "vote_count": int(row["vote_count"]),
            "bayesian_score": round(float(row["bayesian_score"]), 4),
        })
    return results


def get_cold_start_recommendations(user_id: int, n: int = 20) -> list[dict]:
    """Full cold-start path: top Bayesian movies excluding user's watched list.

    Args:
        user_id: The user ID (may have 0 ratings = brand new user).
        n: How many candidates to return.

    Returns:
        List of top-N recommended movies sorted by Bayesian score.
    """
    watched = get_user_watched_ids(user_id)
    return get_top_bayesian(n=n, exclude_ids=watched, min_votes=10)


def sensitivity_analysis() -> dict[str, dict]:
    """Compare m ∈ {5, 10, 30} — used in evaluation report.

    Returns:
        Dict with keys '5', '10', '30' and values containing:
            - eligible_count: # movies with vote_count >= m
            - eligible_pct: % of total catalog eligible
            - top10_titles: list of top-10 movie titles
    """
    stats = build_rating_stats()
    movies_lookup = build_movie_lookup()
    total = len(stats)
    results = {}

    for m in [5, 10, 30]:
        eligible = stats[stats["vote_count"] >= m].copy()
        eligible["w_score"] = eligible.apply(
            lambda r: bayesian_score(r["vote_count"], r["avg_rating"], m=m),
            axis=1,
        )
        top10 = eligible.sort_values("w_score", ascending=False).head(10)
        titles = [movies_lookup.get(int(mid), {}).get("title", str(mid)) for mid in top10["movieId"]]
        results[str(m)] = {
            "eligible_count": int(len(eligible)),
            "eligible_pct": round(len(eligible) / total * 100, 1),
            "top10_titles": titles,
        }
    return results


if __name__ == "__main__":
    print("=== Bayesian Engine — Self-test ===")
    recs = get_cold_start_recommendations(user_id=999, n=5)
    for r in recs:
        print(f"  [{r['bayesian_score']:.3f}] {r['title']} ({r['vote_count']} votes, avg {r['avg_rating']}★)")

    print("\n=== Sensitivity Analysis m ∈ {5, 10, 30} ===")
    sa = sensitivity_analysis()
    for m, info in sa.items():
        print(f"  m={m}: {info['eligible_count']:,} eligible ({info['eligible_pct']}%)")
        print(f"         Top-3: {info['top10_titles'][:3]}")
