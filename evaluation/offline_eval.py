"""
offline_eval.py — Leave-One-Out (LOO) Evaluation + Hit@K metric.

Method:
    For each evaluation user [1, 15, 30]:
        1. Find the movie with the LATEST timestamp (hidden movie).
        2. Remove it from the user's known ratings.
        3. Run the recommendation engine (pretend it's unseen).
        4. Check if the hidden movie appears in Top-K results.

Metrics reported:
    - Hit@K for K ∈ {5, 10, 20}
    - Rank of hidden movie (if found)
    - Fallback path used (bayesian / content / hybrid)

Baseline comparison also included (Popularity + Genre-only).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    build_movie_lookup,
    build_rating_stats,
    get_user_ratings,
    get_user_top_genres,
    load_ratings,
)
from hybrid_engine import get_recommendations

EVAL_USERS = [1, 15, 30]
K_VALUES = [5, 10, 20]


# ---------------------------------------------------------------------------
# LOO Evaluation
# ---------------------------------------------------------------------------

def run_loo_evaluation(user_id: int) -> dict:
    """Run Leave-One-Out evaluation for a single user.

    Returns:
        Dict with hit_at_k, hidden_movie_info, rank, path_used.
    """
    user_ratings = get_user_ratings(user_id)

    if user_ratings.empty:
        return {"user_id": user_id, "error": "no ratings", "hit_at": {}}

    # Find the hidden movie (most recent rating by timestamp)
    hidden_row = user_ratings.iloc[0]  # sorted newest first by get_user_ratings
    hidden_movie_id = int(hidden_row["movieId"])
    hidden_rating = float(hidden_row["rating"])

    movies_lookup = build_movie_lookup()
    hidden_title = movies_lookup.get(hidden_movie_id, {}).get("title", f"Movie {hidden_movie_id}")

    # Simulate the user WITHOUT the hidden movie by using the watched list minus the hidden movie
    # The engine uses get_user_watched_ids — we need to override by passing exclude_ids
    # We actually get recommendations excluding all watched movies, so the hidden movie
    # (if the user has already seen it) would normally be excluded.
    # For LOO: we want to see if the system WOULD have recommended it if unseen.
    # Strategy: exclude_ids = watched_set - {hidden_movie_id}
    from data_loader import get_user_watched_ids
    watched = get_user_watched_ids(user_id)
    loo_exclude = watched - {hidden_movie_id}

    recs, path_used = get_recommendations(user_id, n=20, exclude_ids=loo_exclude)

    rec_movie_ids = [r["movieId"] for r in recs]

    hit_at_k = {}
    rank = None
    for k in K_VALUES:
        top_k_ids = rec_movie_ids[:k]
        hit = hidden_movie_id in top_k_ids
        hit_at_k[f"Hit@{k}"] = int(hit)
        if hit and rank is None:
            rank = top_k_ids.index(hidden_movie_id) + 1

    return {
        "user_id": user_id,
        "hidden_movie_id": hidden_movie_id,
        "hidden_movie_title": hidden_title,
        "hidden_movie_rating": hidden_rating,
        "path_used": path_used,
        "hit_at_k": hit_at_k,
        "rank_in_top20": rank,
        "total_recs": len(recs),
    }


# ---------------------------------------------------------------------------
# Popularity Baseline
# ---------------------------------------------------------------------------

def run_popularity_baseline(user_id: int, k_values=None) -> dict:
    """Baseline: recommend top-K by Bayesian score (no personalization)."""
    if k_values is None:
        k_values = K_VALUES

    user_ratings = get_user_ratings(user_id)
    if user_ratings.empty:
        return {"user_id": user_id, "error": "no ratings", "hit_at_k": {}}

    hidden_movie_id = int(user_ratings.iloc[0]["movieId"])

    from bayesian_engine import get_top_bayesian
    from data_loader import get_user_watched_ids
    watched = get_user_watched_ids(user_id)
    loo_exclude = watched - {hidden_movie_id}

    baseline_recs = get_top_bayesian(n=max(k_values), exclude_ids=loo_exclude)
    baseline_ids = [r["movieId"] for r in baseline_recs]

    hit_at_k = {}
    for k in k_values:
        hit_at_k[f"Hit@{k}"] = int(hidden_movie_id in baseline_ids[:k])

    return {"user_id": user_id, "baseline": "popularity", "hit_at_k": hit_at_k}


# ---------------------------------------------------------------------------
# Genre-only Baseline
# ---------------------------------------------------------------------------

def run_genre_baseline(user_id: int, k_values=None) -> dict:
    """Baseline: recommend top-K in user's top genre by avg rating."""
    if k_values is None:
        k_values = K_VALUES

    user_ratings = get_user_ratings(user_id)
    if user_ratings.empty:
        return {"user_id": user_id, "error": "no ratings", "hit_at_k": {}}

    hidden_movie_id = int(user_ratings.iloc[0]["movieId"])

    top_genres = get_user_top_genres(user_id, top_n=1)
    if not top_genres:
        return {"user_id": user_id, "error": "no genres", "hit_at_k": {}}

    from bayesian_engine import get_top_bayesian
    from data_loader import get_user_watched_ids
    watched = get_user_watched_ids(user_id)
    loo_exclude = watched - {hidden_movie_id}

    genre_recs = get_top_bayesian(n=max(k_values), exclude_ids=loo_exclude, genre_filter=top_genres)
    genre_ids = [r["movieId"] for r in genre_recs]

    hit_at_k = {}
    for k in k_values:
        hit_at_k[f"Hit@{k}"] = int(hidden_movie_id in genre_ids[:k])

    return {"user_id": user_id, "baseline": "genre_only", "hit_at_k": hit_at_k}


# ---------------------------------------------------------------------------
# Diversity Metrics
# ---------------------------------------------------------------------------

def compute_diversity_metrics(user_id: int) -> dict:
    """Compute intra-list diversity and genre coverage for top-20 recs."""
    recs, _ = get_recommendations(user_id, n=20)

    # Collect all genres in top-20
    genre_sets = []
    genre_counts: dict[str, int] = {}
    for m in recs:
        genres = [g for g in m.get("genres", "").split("|") if g and g != "(no genres listed)"]
        genre_sets.append(set(genres))
        for g in genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1

    # Intra-list diversity: avg pairwise Jaccard dissimilarity
    n = len(genre_sets)
    if n < 2:
        diversity = 0.0
    else:
        total_dissim = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                union = genre_sets[i] | genre_sets[j]
                inter = genre_sets[i] & genre_sets[j]
                if union:
                    total_dissim += 1 - len(inter) / len(union)
                    count += 1
        diversity = total_dissim / count if count > 0 else 0.0

    # Genre cap check: any genre exceed max_per_genre=5?
    max_genre_count = max(genre_counts.values()) if genre_counts else 0
    genre_cap_respected = max_genre_count <= 5

    # Top-3 genre diversity
    top3_genres = set()
    for m in recs[:3]:
        for g in m.get("genres", "").split("|"):
            if g and g != "(no genres listed)":
                top3_genres.add(g)

    return {
        "user_id": user_id,
        "intra_list_diversity": round(diversity, 3),
        "unique_genres_top20": len(genre_counts),
        "max_single_genre_count": max_genre_count,
        "genre_cap_respected": genre_cap_respected,
        "unique_genres_top3": len(top3_genres),
        "top3_genre_diverse": len(top3_genres) >= 2,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TrustedAI Offline Evaluation — Leave-One-Out + Hit@K")
    print("=" * 70)
    print("\nPre-computing matrices (may take 10-30 seconds)...\n")

    # LOO Evaluation
    loo_results = []
    for uid in EVAL_USERS:
        print(f"  Running LOO for User {uid}...")
        result = run_loo_evaluation(uid)
        loo_results.append(result)
        title = result.get("hidden_movie_title", "N/A")
        rating = result.get("hidden_movie_rating", "?")
        path = result.get("path_used", "?")
        hits = result.get("hit_at_k", {})
        print(f"    Hidden movie: {title} ({rating}★) | path: {path}")
        print(f"    Hit@5={hits.get('Hit@5', 0)}  Hit@10={hits.get('Hit@10', 0)}  Hit@20={hits.get('Hit@20', 0)}")
        if result.get("rank_in_top20"):
            print(f"    Rank in Top-20: #{result['rank_in_top20']}")
        print()

    # Summary table
    print("\n--- LOO Summary ---")
    print(f"{'User':<8} {'Path':<12} {'Hit@5':<8} {'Hit@10':<8} {'Hit@20':<8} {'Rank':<8}")
    print("-" * 55)
    for r in loo_results:
        rank = r.get("rank_in_top20", "—")
        hits = r.get("hit_at_k", {})
        print(f"{r['user_id']:<8} {r.get('path_used','?'):<12} "
              f"{hits.get('Hit@5', 0):<8} {hits.get('Hit@10', 0):<8} "
              f"{hits.get('Hit@20', 0):<8} {rank!s:<8}")

    # Aggregate Hit@K
    for k in K_VALUES:
        key = f"Hit@{k}"
        avg = sum(r.get("hit_at_k", {}).get(key, 0) for r in loo_results) / len(loo_results)
        print(f"\nAggregate {key}: {avg:.2f} (over {len(loo_results)} users)")

    # Baseline comparison
    print("\n" + "=" * 70)
    print("Baseline Comparison")
    print("=" * 70)
    for uid in EVAL_USERS:
        pop = run_popularity_baseline(uid)
        genre = run_genre_baseline(uid)
        hybrid = next(r for r in loo_results if r["user_id"] == uid)
        print(f"\nUser {uid}:")
        print(f"  Hybrid  : {hybrid.get('hit_at_k', {})}")
        print(f"  Popularity: {pop.get('hit_at_k', {})}")
        print(f"  Genre-only: {genre.get('hit_at_k', {})}")

    # Diversity metrics
    print("\n" + "=" * 70)
    print("Diversity & Coverage Metrics")
    print("=" * 70)
    for uid in EVAL_USERS:
        div = compute_diversity_metrics(uid)
        print(f"\nUser {uid}:")
        for k, v in div.items():
            if k != "user_id":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
