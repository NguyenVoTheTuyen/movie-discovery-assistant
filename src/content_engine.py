"""
content_engine.py — TF-IDF Content Similarity Engine.

Responsibilities:
    1. Build a TF-IDF feature matrix over all movies (plot + genres + tags).
    2. Support query-to-movie similarity (Dạng 2: content search).
    3. Support profile-to-movie similarity (used by hybrid engine for < 20 ratings path).

Design notes:
    - 80% of movies have NO tags (only 1,026 / 5,135 have tags).
      → Dynamic fallback: if tags absent, feature = [plot + genres] only.
    - Plot snippet: first 400 chars used in LLM payload (kept full here for TF-IDF quality).
    - TF-IDF chosen over SentenceTransformers to avoid 500MB model download and
      maintain zero-infra dependency (sufficient for 5K movies in under 4h).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import (
    build_movie_lookup,
    build_rating_stats,
    get_user_watched_ids,
    get_user_ratings,
    get_user_top_genres,
    load_movies,
)

# ---------------------------------------------------------------------------
# Feature text builder
# ---------------------------------------------------------------------------

def _build_feature_text(movie: dict) -> str:
    """Build a single text string representing a movie's content features.

    Dynamic fallback: genres + plot always included; tags added only when present.
    The genres are repeated to boost their weight relative to the long plot text.
    """
    genres_str = movie.get("genres", "")
    if genres_str == "(no genres listed)":
        genres_str = ""

    # Genres get 3x weight by repetition
    genre_tokens = " ".join(genres_str.replace("|", " ").split()) * 3

    plot = movie.get("plot", "")

    tags = movie.get("tags", [])
    tag_tokens = " ".join(tags) if tags else ""

    parts = [genre_tokens, plot, tag_tokens]
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# TF-IDF Matrix — built once, cached
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_tfidf_data() -> tuple[TfidfVectorizer, np.ndarray, list[int]]:
    """Build and cache the TF-IDF matrix.

    Returns:
        (vectorizer, tfidf_matrix, movie_id_list)
        where tfidf_matrix[i] corresponds to movie_id_list[i].
    """
    movies_lookup = build_movie_lookup()
    movie_ids = sorted(movies_lookup.keys())

    corpus = [_build_feature_text(movies_lookup[mid]) for mid in movie_ids]

    vectorizer = TfidfVectorizer(
        max_features=15_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix.toarray(), movie_ids


def _movie_id_to_index(movie_id: int) -> Optional[int]:
    """Return the row index of a movie in the TF-IDF matrix."""
    _, _, movie_ids = _get_tfidf_data()
    try:
        return movie_ids.index(movie_id)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Query → Movie similarity (Dạng 2: content search)
# ---------------------------------------------------------------------------

def search_by_content(
    query: str,
    n: int = 20,
    exclude_ids: Optional[set[int]] = None,
    min_avg_rating: float = 3.5,
    min_votes: int = 10,
) -> list[dict]:
    """Find movies most similar to a free-text query.

    Scoring formula (Dạng 2):
        Score(m) = 0.6 × ContentSim(m, query) + 0.4 × BayesianRating(m)

    Args:
        query: User's natural language query.
        n: Number of results to return.
        exclude_ids: Movie IDs to exclude from results.
        min_avg_rating: Minimum average rating threshold.
        min_votes: Minimum number of votes.

    Returns:
        Ranked list of movie dicts.
    """
    exclude_ids = exclude_ids or set()
    vectorizer, matrix, movie_ids = _get_tfidf_data()
    movies_lookup = build_movie_lookup()
    stats = build_rating_stats().set_index("movieId")

    query_vec = vectorizer.transform([query]).toarray()
    sims = cosine_similarity(query_vec, matrix)[0]  # shape: (n_movies,)

    # Normalize Bayesian scores to [0, 1]
    all_bayesian = stats["bayesian_score"].values
    b_min, b_max = all_bayesian.min(), all_bayesian.max()
    b_range = b_max - b_min if b_max > b_min else 1.0

    results = []
    for idx, movie_id in enumerate(movie_ids):
        if movie_id in exclude_ids:
            continue
        if movie_id not in stats.index:
            continue

        row = stats.loc[movie_id]
        if row["vote_count"] < min_votes or row["avg_rating"] < min_avg_rating:
            continue

        content_sim = float(sims[idx])
        bayesian_norm = float((row["bayesian_score"] - b_min) / b_range)
        final_score = 0.6 * content_sim + 0.4 * bayesian_norm

        movie = movies_lookup[movie_id]
        results.append({
            "movieId": movie_id,
            "title": movie["title"],
            "year": movie.get("year", ""),
            "genres": movie["genres"],
            "plot": movie["plot"],
            "tags": movie["tags"],
            "avg_rating": round(float(row["avg_rating"]), 2),
            "vote_count": int(row["vote_count"]),
            "bayesian_score": round(float(row["bayesian_score"]), 4),
            "content_sim": round(content_sim, 4),
            "final_score": round(final_score, 4),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:n]


# ---------------------------------------------------------------------------
# User Profile → Movie similarity (used in hybrid engine for < 20 ratings)
# ---------------------------------------------------------------------------

def build_user_content_profile(user_id: int) -> Optional[np.ndarray]:
    """Build a TF-IDF profile vector for a user based on their highly-rated movies.

    The profile is the weighted average of TF-IDF vectors of liked movies,
    weighted by the user's rating for each movie.

    Returns:
        Profile vector (1D numpy array) or None if user has no positive ratings.
    """
    vectorizer, matrix, movie_ids = _get_tfidf_data()
    user_ratings = get_user_ratings(user_id)

    if user_ratings.empty:
        return None

    # Consider only positively-rated movies (>= 3.5)
    positive = user_ratings[user_ratings["rating"] >= 3.5]
    if positive.empty:
        return None

    movie_id_set = set(movie_ids)
    profile = np.zeros(matrix.shape[1], dtype=float)
    weight_sum = 0.0

    for _, row in positive.iterrows():
        mid = int(row["movieId"])
        if mid not in movie_id_set:
            continue
        idx = movie_ids.index(mid)
        w = float(row["rating"])
        profile += w * matrix[idx]
        weight_sum += w

    if weight_sum == 0:
        return None

    profile /= weight_sum
    return profile


def get_content_recommendations(
    user_id: int,
    n: int = 20,
    exclude_ids: Optional[set[int]] = None,
    min_avg_rating: float = 3.0,
    min_votes: int = 5,
) -> list[dict]:
    """Return top-N content-based recommendations for a user (< 20 ratings path).

    Args:
        user_id: Target user.
        n: Number of candidates to return.
        exclude_ids: Already-watched movie IDs.
        min_avg_rating: Minimum average community rating.
        min_votes: Minimum number of community votes.

    Returns:
        Ranked list of movie dicts with content_sim score.
    """
    exclude_ids = exclude_ids or set()
    profile = build_user_content_profile(user_id)

    if profile is None:
        # No positive ratings — fall back to genre-only matching
        genres = get_user_top_genres(user_id, top_n=3)
        genre_query = " ".join(genres) + " movie"
        return search_by_content(genre_query, n=n, exclude_ids=exclude_ids,
                                  min_avg_rating=min_avg_rating, min_votes=min_votes)

    _, matrix, movie_ids = _get_tfidf_data()
    movies_lookup = build_movie_lookup()
    stats = build_rating_stats().set_index("movieId")

    # Compute cosine similarity of profile with all movies
    profile_vec = profile.reshape(1, -1)
    sims = cosine_similarity(profile_vec, matrix)[0]

    results = []
    for idx, movie_id in enumerate(movie_ids):
        if movie_id in exclude_ids:
            continue
        if movie_id not in stats.index:
            continue

        row = stats.loc[movie_id]
        if row["vote_count"] < min_votes or row["avg_rating"] < min_avg_rating:
            continue

        movie = movies_lookup[movie_id]
        results.append({
            "movieId": movie_id,
            "title": movie["title"],
            "genres": movie["genres"],
            "plot": movie["plot"],
            "tags": movie["tags"],
            "avg_rating": round(float(row["avg_rating"]), 2),
            "vote_count": int(row["vote_count"]),
            "bayesian_score": round(float(row["bayesian_score"]), 4),
            "content_sim": round(float(sims[idx]), 4),
            "final_score": round(float(sims[idx]), 4),  # content engine: raw sim
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:n]


def compute_movie_content_similarity(movie_id_a: int, movie_id_b: int) -> float:
    """Compute content similarity between two movies."""
    _, matrix, movie_ids = _get_tfidf_data()
    idx_a = _movie_id_to_index(movie_id_a)
    idx_b = _movie_id_to_index(movie_id_b)
    if idx_a is None or idx_b is None:
        return 0.0
    return float(cosine_similarity(matrix[idx_a:idx_a+1], matrix[idx_b:idx_b+1])[0][0])


if __name__ == "__main__":
    print("=== Content Engine — Self-test ===")
    print("Building TF-IDF matrix (first-time, may take a few seconds)...")
    vectorizer, matrix, movie_ids = _get_tfidf_data()
    print(f"✅ Matrix shape: {matrix.shape}")

    print("\n--- Query: 'dark psychological thriller with a twist' ---")
    results = search_by_content("dark psychological thriller with a twist", n=5)
    for r in results:
        print(f"  [{r['final_score']:.3f}] {r['title']} | {r['genres']} | tags: {r['tags'][:3]}")
