"""
cf_engine.py — Collaborative Filtering Engine (User-based Cosine KNN).

Critical fixes (v4):
    1. fillna(0) AFTER mean-centering — NaN = "not seen" = neutral after centering.
       Without this fix, cosine_similarity() returns NaN → entire CF engine breaks.
    2. T_half floor is handled in hybrid_engine.py (recency computation).

Algorithm:
    1. Build user-item matrix (610 users × 5135 movies).
    2. Mean-center each user's ratings (remove individual rating bias).
    3. Fill remaining NaN with 0 (not seen = no deviation from personal mean).
    4. Compute cosine similarity between all user pairs.
    5. For target user: find K=20 nearest neighbors with min_overlap=5 and min_sim=0.1.
    6. For each candidate movie: weighted-average peer ratings → CF Score.

Hyperparameters:
    K             = 20 (nearest neighbors)
    min_overlap   = 5 movies in common
    min_sim       = 0.1 (discard near-zero / negative similarity)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import (
    build_movie_lookup,
    build_rating_stats,
    get_user_ratings,
    get_user_watched_ids,
    load_ratings,
)

# Hyperparameters
CF_K: int = 20          # Number of nearest neighbors
CF_MIN_OVERLAP: int = 5  # Minimum movies in common to consider a neighbor valid
CF_MIN_SIM: float = 0.1  # Minimum similarity threshold


# ---------------------------------------------------------------------------
# Build User-Item Matrix (cached — precomputed at startup)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _build_user_item_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and cache the mean-centered user-item matrix.

    Returns:
        (raw_matrix, normalized_matrix)
        raw_matrix       : original ratings (NaN where not rated)
        normalized_matrix: mean-centered + fillna(0)  ← CRITICAL FIX v4
    """
    ratings = load_ratings()

    # Pivot: rows=users, cols=movies, values=ratings (NaN where missing)
    raw_matrix = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    )

    # Mean-center: subtract each user's personal mean → remove high/low bias
    user_mean = raw_matrix.mean(axis=1)
    normalized = raw_matrix.subtract(user_mean, axis=0)

    # ⚠️ CRITICAL FIX v4: fill NaN with 0 AFTER mean-centering.
    # NaN means "never rated" = no deviation from personal mean = 0 in normalized space.
    # Without this, cosine_similarity() returns NaN for sparse rows.
    normalized = normalized.fillna(0)

    return raw_matrix, normalized


@lru_cache(maxsize=1)
def _build_similarity_matrix() -> tuple[np.ndarray, list[int]]:
    """Compute and cache the full user-user cosine similarity matrix.

    Returns:
        (similarity_matrix, user_id_list)
        similarity_matrix[i, j] = cosine similarity between user_id_list[i] and user_id_list[j]
    """
    _, normalized = _build_user_item_matrix()
    user_ids = list(normalized.index)
    sim_matrix = cosine_similarity(normalized.values)
    return sim_matrix, user_ids


# ---------------------------------------------------------------------------
# Get nearest neighbors for a user
# ---------------------------------------------------------------------------

def get_neighbors(
    user_id: int,
    k: int = CF_K,
    min_overlap: int = CF_MIN_OVERLAP,
    min_sim: float = CF_MIN_SIM,
) -> list[tuple[int, float]]:
    """Return top-K similar users (neighbor_id, similarity) for a target user.

    Neighbors must satisfy:
        - Watched at least `min_overlap` movies in common
        - Similarity >= `min_sim`

    Args:
        user_id: Target user.
        k: Maximum number of neighbors.
        min_overlap: Minimum shared movies.
        min_sim: Minimum similarity score.

    Returns:
        Sorted list of (neighbor_user_id, similarity) descending by similarity.
    """
    raw_matrix, normalized = _build_user_item_matrix()
    sim_matrix, user_ids = _build_similarity_matrix()

    if user_id not in user_ids:
        return []

    target_idx = user_ids.index(user_id)
    target_sims = sim_matrix[target_idx]

    # Get movies rated by the target user
    target_rated = set(raw_matrix.loc[user_id].dropna().index.tolist())

    neighbors: list[tuple[int, float]] = []
    for idx, uid in enumerate(user_ids):
        if uid == user_id:
            continue
        sim = float(target_sims[idx])
        if sim < min_sim:
            continue

        # Overlap check: count movies both users rated
        neighbor_rated = set(raw_matrix.loc[uid].dropna().index.tolist())
        overlap = len(target_rated & neighbor_rated)
        if overlap < min_overlap:
            continue

        neighbors.append((uid, sim))

    # Sort by similarity descending, take top-k
    neighbors.sort(key=lambda x: x[1], reverse=True)
    return neighbors[:k]


# ---------------------------------------------------------------------------
# Compute CF Score for candidate movies
# ---------------------------------------------------------------------------

def compute_cf_scores(user_id: int, candidate_movie_ids: list[int]) -> dict[int, float]:
    """Compute CF Score for each candidate movie.

    CF Score formula:
        Score_CF(u, m) = Σ(sim(u,v) × r_vm) / Σ|sim(u,v)|
                          v ∈ N_K(u) who rated movie m

    If no neighbor has rated movie m → Score_CF = 0 (use content fallback).

    Args:
        user_id: Target user.
        candidate_movie_ids: Movies to score.

    Returns:
        Dict {movieId: cf_score (0.0 – 5.0)}.
    """
    raw_matrix, _ = _build_user_item_matrix()
    neighbors = get_neighbors(user_id)

    if not neighbors:
        return {mid: 0.0 for mid in candidate_movie_ids}

    scores: dict[int, float] = {}
    for movie_id in candidate_movie_ids:
        weighted_sum = 0.0
        sim_sum = 0.0
        for neighbor_uid, sim in neighbors:
            if movie_id in raw_matrix.columns:
                r = raw_matrix.loc[neighbor_uid, movie_id]
                if not np.isnan(r):
                    weighted_sum += sim * r
                    sim_sum += abs(sim)

        if sim_sum > 0:
            scores[movie_id] = round(weighted_sum / sim_sum, 4)
        else:
            scores[movie_id] = 0.0

    return scores


def get_cf_candidate_movies(user_id: int) -> list[int]:
    """Return list of movie IDs that at least one neighbor of user has rated,
    excluding movies the user has already seen.

    These are the candidates to score via CF.
    """
    raw_matrix, _ = _build_user_item_matrix()
    neighbors = get_neighbors(user_id)

    watched = get_user_watched_ids(user_id)

    candidate_set: set[int] = set()
    for neighbor_uid, _ in neighbors:
        neighbor_rated = set(raw_matrix.loc[neighbor_uid].dropna().index.tolist())
        candidate_set.update(neighbor_rated)

    return list(candidate_set - watched)


def get_peer_stats_for_movie(user_id: int, movie_id: int) -> dict:
    """Return peer analysis stats for a given movie (used in Dạng 3).

    Returns:
        Dict with keys:
            peer_count     : # neighbors who rated this movie
            total_neighbors: total # of neighbors found
            peer_avg       : average peer rating
            peer_variance  : variance in peer ratings
            bias_index     : peer_avg - global_avg
            peer_adoption  : peer_count / total_neighbors (%)
            global_avg     : community-wide average for this movie
            global_count   : total community ratings for this movie
    """
    from data_loader import build_rating_stats

    raw_matrix, _ = _build_user_item_matrix()
    neighbors = get_neighbors(user_id)

    stats = build_rating_stats().set_index("movieId")
    global_avg = float(stats.loc[movie_id, "avg_rating"]) if movie_id in stats.index else 0.0
    global_count = int(stats.loc[movie_id, "vote_count"]) if movie_id in stats.index else 0

    peer_ratings: list[float] = []
    for neighbor_uid, _ in neighbors:
        if movie_id in raw_matrix.columns:
            r = raw_matrix.loc[neighbor_uid, movie_id]
            if not np.isnan(r):
                peer_ratings.append(float(r))

    if not peer_ratings:
        return {
            "peer_count": 0,
            "total_neighbors": len(neighbors),
            "peer_avg": 0.0,
            "peer_variance": 0.0,
            "bias_index": 0.0,
            "peer_adoption": 0.0,
            "global_avg": global_avg,
            "global_count": global_count,
        }

    peer_avg = sum(peer_ratings) / len(peer_ratings)
    peer_var = sum((r - peer_avg) ** 2 for r in peer_ratings) / len(peer_ratings)

    return {
        "peer_count": len(peer_ratings),
        "total_neighbors": len(neighbors),
        "peer_avg": round(peer_avg, 2),
        "peer_variance": round(peer_var, 2),
        "bias_index": round(peer_avg - global_avg, 2),
        "peer_adoption": round(len(peer_ratings) / len(neighbors) * 100, 1) if neighbors else 0.0,
        "global_avg": round(global_avg, 2),
        "global_count": global_count,
    }


if __name__ == "__main__":
    print("=== CF Engine — Self-test ===")
    print("Pre-computing user-item matrix...")
    raw, norm = _build_user_item_matrix()
    print(f"✅ Raw matrix shape : {raw.shape}")
    print(f"✅ Norm matrix shape: {norm.shape}")
    print(f"✅ NaN count in normalized: {norm.isna().sum().sum()} (should be 0)")

    print("\n--- Neighbors for User 1 ---")
    neighbors = get_neighbors(user_id=1)
    for uid, sim in neighbors[:5]:
        print(f"  User {uid}: sim={sim:.3f}")

    print("\n--- Peer stats for User 1 on Pulp Fiction (movieId=296) ---")
    peer = get_peer_stats_for_movie(user_id=1, movie_id=296)
    for k, v in peer.items():
        print(f"  {k}: {v}")
