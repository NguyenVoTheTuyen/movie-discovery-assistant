"""
data_loader.py — Load & cache DataFrames from the MovieLens dataset.

Data characteristics (verified from actual data):
- movies_with_plots.csv : 5,135 movies, 100% have plot, avg 3,248 chars
- ratings.csv           : 74,064 ratings, 610 users, 2.36% filled
- tags.csv              : 2,440 tags, 1,026 movies have tag (20%)
- Rating scale          : 0.5 – 5.0 (half-star)
- Global mean C         : 3.4947
- Timestamp range       : 1996-03-29 → 2018-09-24
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Resolved data directory (relative to this file's location)
# ---------------------------------------------------------------------------
_EXAM_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _EXAM_DIR / "data" / "ml-latest-small-filtered"

# Verified constants from actual data
GLOBAL_MEAN_RATING: float = 3.4947
TOTAL_USERS: int = 610
TOTAL_MOVIES: int = 5135


# ---------------------------------------------------------------------------
# Public loading functions — each is cached after first call
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_movies() -> pd.DataFrame:
    """Load movies_with_plots.csv.

    Returns DataFrame with columns:
        movieId (int), title (str), genres (str), plot (str)
    """
    path = DATA_DIR / "movies_with_plots.csv"
    df = pd.read_csv(path, dtype={"movieId": int})
    df["plot"] = df["plot"].fillna("")
    df["genres"] = df["genres"].fillna("(no genres listed)")
    return df


@lru_cache(maxsize=1)
def load_ratings() -> pd.DataFrame:
    """Load ratings.csv.

    Returns DataFrame with columns:
        userId (int), movieId (int), rating (float), timestamp (int)
    """
    path = DATA_DIR / "ratings.csv"
    df = pd.read_csv(path, dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int})
    return df


@lru_cache(maxsize=1)
def load_tags() -> pd.DataFrame:
    """Load tags.csv.

    Returns DataFrame with columns:
        userId (int), movieId (int), tag (str), timestamp (int)
    """
    path = DATA_DIR / "tags.csv"
    df = pd.read_csv(path, dtype={"userId": int, "movieId": int, "timestamp": int})
    df["tag"] = df["tag"].fillna("").str.lower().str.strip()
    return df


@lru_cache(maxsize=1)
def load_links() -> pd.DataFrame:
    """Load links.csv (imdbId, tmdbId mapping)."""
    path = DATA_DIR / "links.csv"
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def build_movie_tags_map() -> dict[int, list[str]]:
    """Return {movieId: [tag1, tag2, ...]} using aggregated tags."""
    tags = load_tags()
    result: dict[int, list[str]] = {}
    for movie_id, group in tags.groupby("movieId"):
        result[int(movie_id)] = group["tag"].dropna().unique().tolist()
    return result


@lru_cache(maxsize=1)
def build_movie_lookup() -> dict[int, dict]:
    """Return {movieId: {title, genres, plot, tags}} for fast O(1) access."""
    movies = load_movies()
    tags_map = build_movie_tags_map()
    lookup: dict[int, dict] = {}
    for _, row in movies.iterrows():
        mid = int(row["movieId"])
        y_val = row.get("year")
        year_str = int(y_val) if pd.notna(y_val) and str(y_val).replace(".0","").isdigit() else ""
        lookup[mid] = {
            "movieId": mid,
            "title": row["title"],
            "year": year_str,
            "genres": row["genres"],
            "plot": row["plot"],
            "tags": tags_map.get(mid, []),
        }
    return lookup


@lru_cache(maxsize=1)
def build_rating_stats() -> pd.DataFrame:
    """Return per-movie rating statistics: count, mean, bayesian_score."""
    ratings = load_ratings()
    stats = ratings.groupby("movieId").agg(
        vote_count=("rating", "count"),
        avg_rating=("rating", "mean"),
    ).reset_index()

    # Bayesian Weighted Rating — m=10, C=3.4947
    m = 10
    C = GLOBAL_MEAN_RATING
    stats["bayesian_score"] = (
        (stats["vote_count"] / (stats["vote_count"] + m)) * stats["avg_rating"]
        + (m / (stats["vote_count"] + m)) * C
    )
    return stats


@lru_cache(maxsize=1)
def get_user_rating_count() -> dict[int, int]:
    """Return {userId: number_of_ratings}."""
    ratings = load_ratings()
    return ratings.groupby("userId")["rating"].count().to_dict()


def get_user_ratings(user_id: int) -> pd.DataFrame:
    """Return all ratings for a specific user, sorted newest first."""
    ratings = load_ratings()
    user_ratings = ratings[ratings["userId"] == user_id].copy()
    user_ratings = user_ratings.sort_values("timestamp", ascending=False)
    return user_ratings


def get_user_watched_ids(user_id: int) -> set[int]:
    """Return the set of movieIds already seen by this user."""
    user_ratings = get_user_ratings(user_id)
    return set(user_ratings["movieId"].tolist())


def get_user_top_genres(user_id: int, top_n: int = 5) -> list[str]:
    """Return the top N genres (by number of high-rated movies) for a user."""
    movies = load_movies()
    user_ratings = get_user_ratings(user_id)

    if user_ratings.empty:
        return []

    # Merge with movie metadata
    merged = user_ratings.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    merged = merged[merged["rating"] >= 3.5]  # positive ratings only

    genre_counts: dict[str, int] = {}
    for genres_str in merged["genres"].dropna():
        if genres_str == "(no genres listed)":
            continue
        for g in genres_str.split("|"):
            genre_counts[g] = genre_counts.get(g, 0) + 1

    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    return [g for g, _ in sorted_genres[:top_n]]


def get_user_top_rated_movies(user_id: int, min_rating: float = 4.5, top_n: int = 5) -> list[dict]:
    """Return top N movies rated >= min_rating for a user."""
    movies_lookup = build_movie_lookup()
    user_ratings = get_user_ratings(user_id)
    high_rated = user_ratings[user_ratings["rating"] >= min_rating].head(top_n)

    result = []
    for _, row in high_rated.iterrows():
        mid = int(row["movieId"])
        if mid in movies_lookup:
            result.append({
                "movieId": mid,
                "title": movies_lookup[mid]["title"],
                "rating": row["rating"],
            })
    return result


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def verify_data_loaded() -> None:
    """Print a summary to verify all datasets loaded correctly."""
    movies = load_movies()
    ratings = load_ratings()
    tags = load_tags()
    stats = build_rating_stats()

    print(f"✅ Movies : {len(movies):,} rows | columns: {list(movies.columns)}")
    print(f"✅ Ratings: {len(ratings):,} rows | users: {ratings['userId'].nunique()} | movies: {ratings['movieId'].nunique()}")
    print(f"✅ Tags   : {len(tags):,} rows | movies with tags: {tags['movieId'].nunique()}")
    print(f"✅ Rating stats computed | eligible (vote>=10): {len(stats[stats['vote_count']>=10]):,}")
    print(f"✅ Global mean C = {ratings['rating'].mean():.4f}")


if __name__ == "__main__":
    verify_data_loaded()
