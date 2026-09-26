"""
web_app.py — FastAPI web server for TrustedAI Movie Discovery Assistant.

Usage:
    python src/web_app.py
    # Then open http://127.0.0.1:8000 in browser

Endpoints:
    GET  /                        — Serve the web UI (index.html)
    POST /api/chat                — Send a message, get AI response
    GET  /api/profile/{user_id}   — Get user profile info
    POST /api/reset/{user_id}     — Reset conversation session
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data_loader import (
    get_user_rating_count,
    get_user_top_genres,
    get_user_top_rated_movies,
    verify_data_loaded,
)
from session_manager import session_manager

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        verify_data_loaded()
        print("✅ Dataset loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
    yield

app = FastAPI(title="TrustedAI Movie Assistant", version="1.0.0", lifespan=lifespan)

@app.get("/health")
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}



# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_id: int
    message: str
    offline: bool = False


class ChatResponse(BaseModel):
    intent: str
    confidence: str
    path_used: str
    response: str
    elapsed_ms: int


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Process a user query and return the AI response."""
    from assistant_agent import process_query

    t0 = time.perf_counter()
    try:
        result = process_query(req.user_id, req.message, offline=req.offline)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return ChatResponse(
        intent=result["intent"],
        confidence=result["confidence"],
        path_used=result["path_used"],
        response=result["response"],
        elapsed_ms=elapsed_ms,
    )


@app.get("/api/users")
async def get_users():
    """Return all known user IDs with their rating count and active path."""
    rating_counts = get_user_rating_count()
    users_list = []
    for uid in sorted(rating_counts.keys()):
        count = rating_counts[uid]
        if count == 0:
            tier = "Cold-Start"
        elif count < 20:
            tier = "Sparse (<20 ratings)"
        else:
            tier = "Active (≥20 ratings)"
        users_list.append({"id": uid, "ratings": count, "tier": tier})

    if 999 not in rating_counts:
        users_list.append({"id": 999, "ratings": 0, "tier": "Cold-Start Demo"})

    return {"users": users_list}


@app.get("/api/profile/{user_id}")
async def get_profile(user_id: int):
    """Return user profile info."""
    rating_counts = get_user_rating_count()
    count = rating_counts.get(user_id, 0)
    genres = get_user_top_genres(user_id, top_n=5)
    top_rated = get_user_top_rated_movies(user_id, min_rating=4.5, top_n=5)

    if count == 0:
        path = "Cold-Start → Bayesian"
    elif count < 20:
        path = "Content-Based → TF-IDF"
    else:
        path = "Hybrid → CF + Content + Recency"

    return {
        "user_id": user_id,
        "total_ratings": count,
        "active_path": path,
        "top_genres": genres,
        "top_rated_movies": [
            {"title": m["title"], "rating": m["rating"]} for m in top_rated
        ],
    }


@app.post("/api/reset/{user_id}")
async def reset_session(user_id: int):
    """Reset conversation history for a user."""
    session_manager.reset(user_id)
    return {"status": "ok", "message": f"Session for User {user_id} reset."}


# ---------------------------------------------------------------------------
# Serve the HTML UI
# ---------------------------------------------------------------------------

_HTML_PATH = Path(__file__).parent.parent / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if _HTML_PATH.exists():
        return HTMLResponse(content=_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=True)
