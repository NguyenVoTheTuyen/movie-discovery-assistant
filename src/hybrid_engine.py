"""
hybrid_engine.py — Hybrid Scoring Engine (>= 20 ratings path).

Formula:
    Score_hybrid(m) = α × Score_CF(m) + β × Sim_Content(m, profile) + γ × Recency(m)

Weights:
    α = 0.50 (CF Score)
    β = 0.30 (Content Similarity)
    γ = 0.20 (Recency Weight)

Fallback when Score_CF = 0:
    Score_hybrid(m) = 0.60 × Sim_Content(m, profile) + 0.40 × Recency(m)

Key fixes (v4):
    - T_half floor = 30 days (prevents degenerate recency for User 1 & 30
      who rated all movies in a single day).
    - Unknown genre → "Unknown" (prevents KeyError in diversity filter).

Two-Stage Adaptive:
    - 0 ratings   → Bayesian (cold start, bayesian_engine.py)
    - < 20 ratings → Content (content_engine.py)
    - >= 20 ratings → Hybrid (this module)
"""

from __future__ import annotations

import math
import statistics
from typing import Optional

import numpy as np
import pandas as pd

from bayesian_engine import get_cold_start_recommendations
from cf_engine import (
    compute_cf_scores,
    get_cf_candidate_movies,
    get_neighbors,
)
from content_engine import (
    build_user_content_profile,
    get_content_recommendations,
)
from data_loader import (
    build_movie_lookup,
    build_rating_stats,
    get_user_rating_count,
    get_user_ratings,
    get_user_watched_ids,
)

# Hybrid weights
ALPHA: float = 0.50  # CF weight
BETA: float = 0.30   # Content weight
GAMMA: float = 0.20  # Recency weight

# Genre diversity constraint
MAX_PER_GENRE: int = 5
MIN_REPS_FAVORITE_GENRE: int = 2

# T_half constants (seconds)
T_HALF_FLOOR_SECONDS: int = 30 * 24 * 3600    # 30 days (floor)
T_HALF_FALLBACK_SECONDS: int = 180 * 24 * 3600  # 180 days (fallback)


# ---------------------------------------------------------------------------
# Recency Weight — exponential decay with T_half floor fix
# ---------------------------------------------------------------------------

def compute_lambda(user_timestamps: list[int]) -> float:
    """Compute decay constant λ for exponential recency weighting.

    Critical fix (v4): T_half has a floor of 30 days to prevent degenerate
    cases where all ratings were made in the same day (User 1: 190 ratings on
    2000-07-30, User 30: 18 ratings on 2017-07-18).

    Args:
        user_timestamps: Unix timestamps of all ratings by the user.

    Returns:
        λ = ln(2) / T_half   (in units of 1/second)
    """
    if len(user_timestamps) < 5:
        return math.log(2) / T_HALF_FALLBACK_SECONDS

    sorted_ts = sorted(set(user_timestamps))
    deltas = [
        sorted_ts[i + 1] - sorted_ts[i]
        for i in range(len(sorted_ts) - 1)
        if sorted_ts[i + 1] > sorted_ts[i]
    ]

    if not deltas:
        # All ratings on the same timestamp (User 1, User 30)
        return math.log(2) / T_HALF_FALLBACK_SECONDS

    T_half_raw = statistics.median(deltas) * 4
    T_half = max(T_half_raw, T_HALF_FLOOR_SECONDS)  # ← KEY FIX v4
    return math.log(2) / T_half


def compute_recency_weight(user_id: int) -> dict[int, float]:
    """Compute per-movie recency weight for a user using time-decayed scoring.

    Uses t_ref = max(user_timestamps) as the reference point (not t_now=2026)
    to avoid 8-30 year gaps that would make all weights near zero.

    Returns:
        {movieId: recency_weight (0.0 – 1.0)}
    """
    user_ratings = get_user_ratings(user_id)
    if user_ratings.empty:
        return {}

    timestamps = user_ratings["timestamp"].tolist()
    lam = compute_lambda(timestamps)
    t_ref = max(timestamps)

    weights: dict[int, float] = {}
    for _, row in user_ratings.iterrows():
        delta_t = t_ref - int(row["timestamp"])
        weights[int(row["movieId"])] = math.exp(-lam * delta_t)

    # Normalize to [0, 1]
    if weights:
        max_w = max(weights.values())
        if max_w > 0:
            weights = {mid: w / max_w for mid, w in weights.items()}

    return weights


def compute_genre_recency_scores(user_id: int) -> dict[str, float]:
    """Compute time-decayed recency score per genre for a user.

    Implements Plan Section 3.3 formula:
        Score(g) = Σ Rating_i × exp(-λ × (t_ref - t_i)),  i ∈ Movies_of_genre_g

    This allows candidate movies to receive a genre-specific recency signal:
    genres the user has been watching *recently* get a higher weight,
    so candidate movies of that genre rank higher in the hybrid score.

    Returns:
        {genre: normalized_recency_score (0.0 – 1.0)}
        Empty dict if user has no ratings.
    """
    user_ratings = get_user_ratings(user_id)
    if user_ratings.empty:
        return {}

    from data_loader import build_movie_lookup
    movies_lookup = build_movie_lookup()

    timestamps = user_ratings["timestamp"].tolist()
    lam = compute_lambda(timestamps)
    t_ref = max(timestamps)

    genre_scores: dict[str, float] = {}

    for _, row in user_ratings.iterrows():
        mid = int(row["movieId"])
        rating = float(row["rating"])
        delta_t = t_ref - int(row["timestamp"])
        decay = math.exp(-lam * delta_t)
        contribution = rating * decay  # Plan formula: Rating_i × exp(-λ × Δt)

        # Attribute contribution to the movie's primary genre
        movie = movies_lookup.get(mid, {})
        genre = get_primary_genre(movie.get("genres", ""))
        genre_scores[genre] = genre_scores.get(genre, 0.0) + contribution

    # Normalize to [0, 1]
    if genre_scores:
        max_score = max(genre_scores.values())
        if max_score > 0:
            genre_scores = {g: s / max_score for g, s in genre_scores.items()}

    return genre_scores


# ---------------------------------------------------------------------------
# Genre diversity filter
# ---------------------------------------------------------------------------

def get_primary_genre(genres_str: str) -> str:
    """Extract primary genre, handling '(no genres listed)' → 'Unknown'."""
    if not genres_str or genres_str.strip() == "(no genres listed)":
        return "Unknown"
    return genres_str.split("|")[0]


def apply_genre_diversity(
    ranked_candidates: list[dict],
    top_n: int = 20,
    max_per_genre: int = MAX_PER_GENRE,
    min_reps_per_fav_genre: int = MIN_REPS_FAVORITE_GENRE,
    favorite_genres: Optional[list[str]] = None,
) -> list[dict]:
    """Greedy diversity filter: pick top_n movies with genre diversity.

    Algorithm:
        1. From top-50 candidates (by score), greedily pick until top_n.
        2. Constraint: each primary genre can appear at most max_per_genre times.
        3. Guarantee: each favorite genre has >= min_reps_per_fav_genre representatives.
        4. Fallback: if not enough diverse movies, relax constraints and fill by rank.

    Args:
        ranked_candidates: Sorted list of movie dicts (best first).
        top_n: Target number of movies to return.
        max_per_genre: Max movies per genre.
        min_reps_per_fav_genre: Minimum movies for each favorite genre.
        favorite_genres: User's top genres to guarantee representation.

    Returns:
        Filtered & diversified list of up to top_n movies.
    """
    pool = ranked_candidates[:50]  # work on top-50
    fav_set = set(g.lower() for g in (favorite_genres or []))
    genre_counts: dict[str, int] = {}
    selected: list[dict] = []

    # Pass 1: greedy selection with max_per_genre constraint
    for movie in pool:
        if len(selected) >= top_n:
            break
        genre = get_primary_genre(movie.get("genres", ""))
        count = genre_counts.get(genre, 0)
        if count < max_per_genre:
            selected.append(movie)
            genre_counts[genre] = count + 1

    # Pass 2: ensure favorite genres have >= min_reps representatives
    if fav_set:
        selected_ids = {m["movieId"] for m in selected}
        fav_counts = {
            g: sum(1 for m in selected if get_primary_genre(m["genres"]).lower() == g)
            for g in fav_set
        }
        for fav_genre in fav_set:
            while fav_counts.get(fav_genre, 0) < min_reps_per_fav_genre:
                # Find next candidate in pool with this genre not yet selected
                found = False
                for movie in pool:
                    if movie["movieId"] in selected_ids:
                        continue
                    if get_primary_genre(movie["genres"]).lower() == fav_genre:
                        selected.append(movie)
                        selected_ids.add(movie["movieId"])
                        fav_counts[fav_genre] = fav_counts.get(fav_genre, 0) + 1
                        found = True
                        break
                if not found:
                    break  # No more candidates for this genre

    # Fallback: if still < top_n, fill from remaining pool by rank
    if len(selected) < top_n:
        selected_ids = {m["movieId"] for m in selected}
        for movie in pool:
            if movie["movieId"] not in selected_ids:
                selected.append(movie)
                if len(selected) >= top_n:
                    break

    return selected[:top_n]


# ---------------------------------------------------------------------------
# Hybrid Scoring Engine
# ---------------------------------------------------------------------------

def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize a list to [0, 1]."""
    if not values:
        return values
    min_v, max_v = min(values), max(values)
    rng = max_v - min_v
    if rng == 0:
        return [0.5] * len(values)
    return [(v - min_v) / rng for v in values]


def get_hybrid_recommendations(
    user_id: int,
    n: int = 20,
    exclude_ids: Optional[set[int]] = None,
) -> list[dict]:
    """Return top-N hybrid recommendations for a user with >= 20 ratings.

    Args:
        user_id: Target user.
        n: Number of candidates to return.
        exclude_ids: Additional movie IDs to exclude.

    Returns:
        Diversified top-N movie dicts with component scores attached.
    """
    watched = get_user_watched_ids(user_id)
    exclude = watched | (exclude_ids or set())

    movies_lookup = build_movie_lookup()
    stats = build_rating_stats().set_index("movieId")

    # Step 1: Get CF candidate movies (movies neighbor have seen, user hasn't)
    cf_candidates = get_cf_candidate_movies(user_id)
    if not cf_candidates:
        # No CF neighbors → fall back to content
        return get_content_recommendations(user_id, n=n, exclude_ids=exclude)

    # Limit candidates to avoid O(n²) issues
    cf_candidates = cf_candidates[:2000]

    # Step 2: Compute CF scores
    cf_scores = compute_cf_scores(user_id, cf_candidates)

    # Step 3: Build user content profile
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    from content_engine import _get_tfidf_data

    _, matrix, movie_ids = _get_tfidf_data()
    movie_id_set = set(movie_ids)
    profile = build_user_content_profile(user_id)

    content_scores: dict[int, float] = {}
    if profile is not None:
        profile_vec = profile.reshape(1, -1)
        for mid in cf_candidates:
            if mid in movie_id_set:
                idx = movie_ids.index(mid)
                content_scores[mid] = float(cos_sim(profile_vec, matrix[idx:idx+1])[0][0])

    # Step 4: Genre-based recency scores (Plan Section 3.3)
    # Score(g) = Σ Rating_i × exp(-λ × (t_ref - t_i)) for each genre g
    # Each candidate movie inherits the recency score of its primary genre →
    # genres the user watched recently contribute a higher γ signal.
    genre_recency = compute_genre_recency_scores(user_id)
    avg_recency = (sum(genre_recency.values()) / len(genre_recency)) if genre_recency else 0.5

    # Step 5: Build scored candidate list
    raw_cf = [cf_scores.get(mid, 0.0) for mid in cf_candidates]
    raw_content = [content_scores.get(mid, 0.0) for mid in cf_candidates]

    norm_cf = _normalize(raw_cf)
    norm_content = _normalize(raw_content)

    scored: list[dict] = []
    for i, mid in enumerate(cf_candidates):
        if mid in exclude:
            continue
        if mid not in stats.index:
            continue
        if mid not in movies_lookup:
            continue

        row = stats.loc[mid]
        cf_norm = norm_cf[i]
        cont_norm = norm_content[i]

        # Genre-based recency: candidate movie inherits its primary genre's recency score.
        # Fallback to avg_recency if genre not in user's history.
        movie_genre = get_primary_genre(movies_lookup[mid].get("genres", ""))
        rec_norm = genre_recency.get(movie_genre, avg_recency)

        # Use CF fallback weights when CF score is zero
        if cf_norm == 0.0:
            hybrid_score = 0.60 * cont_norm + 0.40 * rec_norm
        else:
            hybrid_score = ALPHA * cf_norm + BETA * cont_norm + GAMMA * rec_norm

        movie = movies_lookup[mid]
        scored.append({
            "movieId": mid,
            "title": movie["title"],
            "year": movie.get("year", ""),
            "genres": movie["genres"],
            "plot": movie["plot"],
            "tags": movie["tags"],
            "avg_rating": round(float(row["avg_rating"]), 2),
            "vote_count": int(row["vote_count"]),
            "bayesian_score": round(float(row["bayesian_score"]), 4),
            "cf_score": round(float(raw_cf[i]), 4),
            "content_sim": round(float(raw_content[i]), 4),
            "hybrid_score": round(float(hybrid_score), 4),
            "final_score": round(float(hybrid_score), 4),
        })

    scored.sort(key=lambda x: x["hybrid_score"], reverse=True)

    # Step 6: Genre diversity filter
    from data_loader import get_user_top_genres
    favorite_genres = get_user_top_genres(user_id, top_n=3)
    diversified = apply_genre_diversity(
        scored,
        top_n=n,
        max_per_genre=MAX_PER_GENRE,
        min_reps_per_fav_genre=MIN_REPS_FAVORITE_GENRE,
        favorite_genres=favorite_genres,
    )
    return diversified


# ---------------------------------------------------------------------------
# Main routing: Two-Stage Adaptive Hybrid
# ---------------------------------------------------------------------------

def get_recommendations(user_id: int, n: int = 20, exclude_ids: Optional[set[int]] = None) -> tuple[list[dict], str]:
    """Route to the appropriate recommendation path based on user history count.

    Returns:
        (recommendations_list, path_used)
        path_used is one of: "bayesian", "content", "hybrid"
    """
    rating_counts = get_user_rating_count()
    count = rating_counts.get(user_id, 0)

    if count == 0:
        recs = get_cold_start_recommendations(user_id, n=n)
        return recs, "bayesian"
    elif count < 20:
        watched = get_user_watched_ids(user_id)
        exclude = watched | (exclude_ids or set())
        recs = get_content_recommendations(user_id, n=n, exclude_ids=exclude)
        return recs, "content"
    else:
        recs = get_hybrid_recommendations(user_id, n=n, exclude_ids=exclude_ids)
        return recs, "hybrid"


if __name__ == "__main__":
    print("=== Hybrid Engine — Self-test ===")

    for test_user in [1, 30, 999]:
        recs, path = get_recommendations(test_user, n=5)
        print(f"\n--- User {test_user} → path: {path} ({len(recs)} recs) ---")
        for r in recs:
            print(f"  [{r.get('final_score', 0):.3f}] {r['title']} | {r['genres']}")
