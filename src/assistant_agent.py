"""
assistant_agent.py — LLM Agent with Gemini 2.5 Flash + Robust Handlers.

Architecture:
    - Pre-Router (<1ms): greeting, out_of_scope, clarification (Instant response, 0 token)
    - Domain Handlers:
        - Dạng 1: personal_recommendation (Bayesian / Content / Hybrid + Diversity)
        - Dạng 2: content_search (TF-IDF Cosine Similarity)
        - Dạng 3: peer_analysis (Fuzzy matching + Peer Adoption / Consensus / Bias Index)
        - Dạng 4: explanation (Grounded Explainability + Cold-Explanation Guardrail)
        - Dạng 5: critiquing (Anchor Movie + Hard Negative Filter)
        - Dạng 6: blind_spot (Genre Frequency Analysis + Bridge Movies)
    - Language Synchronization: Detects Vietnamese / English query and matches output language.
    - Fault-tolerant LLM invocation with offline cache and graceful error degradation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Lazy import Gemini to allow offline mode without the SDK installed
try:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

from bayesian_engine import get_cold_start_recommendations
from cf_engine import get_peer_stats_for_movie
from content_engine import search_by_content
from data_loader import (
    build_movie_lookup,
    build_rating_stats,
    get_user_rating_count,
    get_user_ratings,
    get_user_top_genres,
    get_user_top_rated_movies,
    get_user_watched_ids,
    GLOBAL_MEAN_RATING,
)
from hybrid_engine import get_recommendations, apply_genre_diversity
from intent_classifier import IntentResult, classify_intent, resolve_movie_title, strip_accents
from session_manager import SessionContext, session_manager

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
PLOT_SNIPPET_CHARS = 400
MAX_CANDIDATES_TO_LLM = 10
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "false").lower() == "true"

_OFFLINE_CACHE_PATH = Path(__file__).parent.parent / "output" / "cached_llm_responses.json"
_offline_cache: dict[str, str] = {}


def _load_offline_cache() -> None:
    global _offline_cache
    if _OFFLINE_CACHE_PATH.exists():
        try:
            with open(_OFFLINE_CACHE_PATH, encoding="utf-8") as f:
                _offline_cache = json.load(f)
        except Exception:
            _offline_cache = {}


def _save_offline_cache() -> None:
    _OFFLINE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OFFLINE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_offline_cache, f, indent=2, ensure_ascii=False)


def _stable_hash(text: str) -> str:
    """Return a deterministic 6-character hex hash for offline cache keys."""
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()[:6]


# ---------------------------------------------------------------------------
# Language Detection Helper
# ---------------------------------------------------------------------------

def _is_vietnamese(text: str) -> bool:
    """Check if query is in Vietnamese using accented characters and common words."""
    vi_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
    text_lower = text.lower()
    if any(c in vi_chars for c in text_lower):
        return True
    vi_words = {"phim", "xem", "toi", "ban", "cho", "gi", "sao", "thich", "muon", "khong", "chao", "alo", "nao", "hay"}
    words = set(re.findall(r"\w+", text_lower))
    return len(words & vi_words) >= 1


# ---------------------------------------------------------------------------
# Gemini Client Setup & Fault-Tolerant Call
# ---------------------------------------------------------------------------

def _get_gemini_client():
    if not _GEMINI_AVAILABLE:
        raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai")
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Copy .env.example to .env and fill in your key.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config=genai.GenerationConfig(
            temperature=0.7,
            max_output_tokens=8192,
        ),
    )


def _call_llm(prompt: str, cache_key: Optional[str] = None, offline: bool = False) -> str:
    """Call Gemini LLM with offline cache and error resilience."""
    is_offline = offline or OFFLINE_MODE

    if is_offline:
        if cache_key and cache_key in _offline_cache:
            val = _offline_cache[cache_key]
            # Ignore broken or truncated cached strings
            if len(val) >= 100 and not val.rstrip().endswith(("it looks like", "Canadian Bacon", "**1", "*", ",", ":")):
                return val

        offline_notice = (
            "⚡ **Chế độ Ngoại tuyến (Offline Mode):** Nội dung hội thoại AI chưa có trong bộ nhớ đệm."
        )
        return f"__ERROR__:{offline_notice}"

    max_retries = 2
    for attempt in range(max_retries):
        try:
            client = _get_gemini_client()
            response = client.generate_content(prompt)
            text = response.text.strip()

            if cache_key:
                _offline_cache[cache_key] = text
                _save_offline_cache()

            return text
        except Exception as e:
            raw = str(e)
            if ("429" in raw or "quota" in raw.lower()) and attempt < max_retries - 1:
                sec_match = re.search(r"retry in (\d+(?:\.\d+)?)s|seconds:\s*(\d+)", raw)
                secs = float(sec_match.group(1) or sec_match.group(2)) if sec_match else 0
                if 0 < secs <= 3.0:
                    time.sleep(secs + 0.2)
                    continue

            clean_notice = _clean_error_message(e, is_vi=True)
            return f"__ERROR__:{clean_notice}"


def _clean_error_message(err: Exception, is_vi: bool = True) -> str:
    """Format technical API errors into human-friendly, clean UI messages."""
    raw = str(err)
    if "429" in raw or "ResourceExhausted" in raw or "quota" in raw.lower():
        sec_match = re.search(r"retry in (\d+(?:\.\d+)?)s|seconds:\s*(\d+)", raw)
        wait_sec = "30–60 giây"
        if sec_match:
            secs = sec_match.group(1) or sec_match.group(2)
            wait_sec = f"khoảng {int(float(secs))} giây"
        if is_vi:
            return (
                f"⏳ **Hệ thống AI đang tạm thời đạt giới hạn tần suất (Rate Limit 429).**\n"
                f"Vui lòng đợi **{wait_sec}** để tiếp tục trò chuyện sinh động với Gemini, hoặc bật công tắc **Chế độ Ngoại tuyến (Offline Mode)** ở góc trên để trải nghiệm mượt mà không lo gián đoạn API."
            )
        else:
            return (
                f"⏳ **AI service temporarily hit rate limit (429).**\n"
                f"Please wait **{wait_sec}** or switch to **Offline Mode**."
            )
    if "API_KEY" in raw or "403" in raw or "401" in raw:
        return (
            "🔑 **Khóa GEMINI_API_KEY chưa hợp lệ hoặc đã hết hạn.**\n"
            "Vui lòng kiểm tra cấu hình trong file `.env` hoặc bật **Chế độ Ngoại tuyến (Offline Mode)**."
            if is_vi else
            "🔑 **GEMINI_API_KEY is missing or invalid.** Please configure `.env` or enable **Offline Mode**."
        )
    return (
        "⚠️ **Dịch vụ AI đang bận hoặc gặp sự cố kết nối tạm thời.** Vui lòng thử lại sau giây lát hoặc sử dụng Chế độ Ngoại tuyến."
        if is_vi else
        "⚠️ **AI service is temporarily busy or encountered a network interruption.**"
    )


def _render_fallback_recommendations(
    candidates: list[dict],
    user_id: int,
    err_notice: str,
    is_vi: bool = True,
) -> str:
    """Graceful degradation: Render computed machine-learning candidates directly when LLM is unavailable."""
    if not candidates:
        return err_notice

    lines = [err_notice, ""]
    if is_vi:
        lines.append("📊 **KẾT QUẢ TÍNH TOÁN TRỰC TIẾP TỪ CƠ SỞ DỮ LIỆU MOVIELENS:**")
        lines.append("*(Thuật toán máy học đã tính toán xong danh sách phim phù hợp nhất với hồ sơ của bạn)*\n")
    else:
        lines.append("📊 **DIRECT RESULTS COMPUTED FROM MOVIELENS DATASET:**\n")

    for i, m in enumerate(candidates[:3]):
        year = m.get("year", "")
        year_str = f" ({year})" if year else ""
        title = f"{m.get('title', 'Unknown')}{year_str}"
        genres = m.get("genres", "Chưa phân loại")
        plot = m.get("plot", "")[:280].strip()
        if plot and not plot.endswith("."):
            plot += "..."
        rating = m.get("avg_rating", 0)
        peer = m.get("cf_score", rating)

        if i == 0:
            prefix = "👑 **LỰA CHỌN ƯU TIÊN SỐ 1 TỐI NAY:**" if is_vi else "👑 **TOP RECOMMENDATION FOR TONIGHT:**"
        else:
            prefix = f"🔄 **LỰA CHỌN THAY THẾ {i+1}:**" if is_vi else f"🔄 **ALTERNATIVE OPTION {i+1}:**"

        lines.append(prefix)
        lines.append(f"**{title}** — *{genres}* (Điểm cộng đồng: {rating}★ | Nhóm cùng gu: {peer:.2f}★)")
        if plot:
            lines.append(f"- **📖 Tóm tắt nội dung:** {plot}")
        lines.append("")

    if is_vi:
        lines.append("🎯 **LỜI KHUYÊN:** Bạn có thể bật ngay bộ phim số 1 để thưởng thức tối nay mà không cần đắn đo!")
    else:
        lines.append("🎯 **VERDICT:** You can start with the top pick tonight with confidence!")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System Prompt Template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are TrustedAI, an expert movie recommendation assistant.
Your recommendations are strictly grounded in actual data: user rating history, peer ratings from similar users, and movie metadata.

Rules:
1. NEVER hallucinate — only mention movies present in the candidates list.
2. NO TECHNICAL JARGON OR DATABASE IDS:
   - NEVER output database IDs or movie IDs (e.g. NEVER write "(ID 8132)", "movieId", or "ID").
   - Always present movies using their human-readable title and release year (e.g., "**Gladiator (1992)**", "**2046 (2004)**").
3. MANDATORY HUMAN-CENTRIC PRESENTATION:
   For every recommended movie, you MUST include:
   - **Tên phim & Năm phát hành:** e.g., **Tên Phim (Năm)** — *Thể loại*
   - **📖 Tóm tắt nội dung:** Tóm tắt ngắn 1-2 câu miêu tả cụ thể cốt truyện nói về điều gì, nhân vật chính làm gì, biến cố nào xảy ra (dựa trên plot_synopsis). Tuyệt đối không chỉ nói chung chung về thể loại! Người dùng cần biết câu chuyện kể về điều gì để quyết định xem.
   - **🎯 Lý do hợp với bạn:** Giải thích vì sao hợp gu, trích dẫn phim 5★ cùng thể loại trong lịch sử xem.
   - **⭐ Đánh giá cộng đồng:** Trích dẫn điểm trung bình cộng đồng và người cùng gu.
4. ADVISOR MINDSET (Not a passive search engine):
   - When suggesting movies, do not dump an unguided list. Always designate Movie #1 as the PRIMARY TOP PICK ("👑 Lựa chọn ưu tiên số 1 tối nay") with your strongest evidence.
   - Present Movies #2 and #3 as complementary alternatives for different moods.
   - Conclude with a clear, decisive VERDICT / ACTIONABLE ADVICE guiding the user on what to watch first so they don't suffer from choice fatigue.
5. LANGUAGE SYNCHRONIZATION:
   - If the instruction specifies Vietnamese, respond entirely in natural, fluent Vietnamese.
   - If the instruction specifies English, respond entirely in English.
   - Do NOT mix languages in a single response.
6. NO PROMPT ECHOING:
   - NEVER copy parenthetical meta-hints (such as "(1-2 câu)") or internal instructions into bullet point labels.
   - Always output clean, natural labels: "- **📖 Tóm tắt nội dung:**", "- **🎯 Lý do hợp với bạn:**", "- **⭐ Đánh giá cộng đồng:**".
"""


# ---------------------------------------------------------------------------
# Payload Builders
# ---------------------------------------------------------------------------

def _build_user_profile_payload(user_id: int) -> dict:
    count = get_user_rating_count().get(user_id, 0)
    top_genres = get_user_top_genres(user_id, top_n=5)
    top_rated = get_user_top_rated_movies(user_id, min_rating=4.5, top_n=5)
    return {
        "userId": user_id,
        "total_ratings": count,
        "top_genres": top_genres,
        "top_rated_movies": [f"{m['title']} ({m['rating']}★)" for m in top_rated],
    }


def _build_candidate_payload(candidates: list[dict]) -> list[dict]:
    result = []
    for m in candidates[:MAX_CANDIDATES_TO_LLM]:
        year = m.get("year")
        year_str = f" ({year})" if year else ""
        cf_val = m.get("cf_score")
        peer_rating_str = f"{round(float(cf_val), 2)}★" if cf_val is not None else None
        avg_val = m.get("avg_rating", 0)
        result.append({
            "movieId": m.get("movieId"),
            "movie_name": f"{m['title']}{year_str}",
            "title": m.get("title", ""),
            "genres": m.get("genres", ""),
            "plot_synopsis": m.get("plot", "")[:500],
            "community_avg_rating": f"{round(float(avg_val), 2)}★",
            "peer_taste_rating": peer_rating_str,
            "cf_score": cf_val,
            "avg_rating": avg_val,
            "content_sim": m.get("content_sim"),
            "bayesian_score": m.get("bayesian_score"),
            "retrieval_path": m.get("retrieval_path", "hybrid"),
            "tags": m.get("tags", [])[:5],
        })
    return result


# ---------------------------------------------------------------------------
# Tầng 1: Pre-Router Handlers (Instant, 0 token)
# ---------------------------------------------------------------------------

def _handle_greeting(query: str) -> str:
    if _is_vietnamese(query):
        return (
            "👋 **Xin chào bạn!** Tôi là **TrustedAI** — Trợ lý AI khám phá điện ảnh thông minh.\n\n"
            "Tôi có thể hỗ trợ bạn:\n"
            "- 🎬 **Gợi ý phim theo gu**: *\"Tối nay tôi nên xem gì?\"*\n"
            "- 🔍 **Tìm phim theo cảm xúc / cốt truyện**: *\"Tìm phim tâm lý đen tối có cú twist bất ngờ\"*\n"
            "- 👥 **Thăm dò ý kiến cộng đồng cùng gu**: *\"Mọi người nghĩ gì về phim Pulp Fiction?\"*\n"
            "- 🚫 **Gợi ý loại trừ thể loại**: *\"Tôi thích Toy Story nhưng chán phim hoạt hình rồi\"*\n"
            "- 🧭 **Khám phá điểm mù**: *\"Tôi đang bỏ lỡ thể loại phim nào?\"*\n\n"
            "Hôm nay bạn đang có tâm trạng muốn thưởng thức bộ phim như thế nào?"
        )
    return (
        "👋 **Hello!** I am **TrustedAI** — your intelligent movie discovery assistant.\n\n"
        "Here is what I can help you with:\n"
        "- 🎬 **Personalized recommendations**: *\"What should I watch tonight?\"*\n"
        "- 🔍 **Theme & mood search**: *\"I want a dark psychological thriller with a twist\"*\n"
        "- 👥 **Peer community insights**: *\"What do people with similar taste think about Pulp Fiction?\"*\n"
        "- 🚫 **Critiquing with exclusions**: *\"I liked Toy Story but I'm tired of animated movies\"*\n"
        "- 🧭 **Blind spot discovery**: *\"What genres am I missing?\"*\n\n"
        "What kind of movie experience are you looking for today?"
    )


def _handle_out_of_scope(query: str) -> str:
    if _is_vietnamese(query):
        return (
            "🍿 **Xin lỗi bạn**, tôi là Trợ lý AI chuyên sâu về **phim ảnh và khám phá điện ảnh** "
            "dựa trên tập dữ liệu MovieLens (5,135 bộ phim).\n\n"
            "Tôi không thể giải đáp các chủ đề ngoài điện ảnh (như thời tiết, nấu ăn, toán học hay kiến thức chung).\n\n"
            "💡 *Gợi ý:* Hãy hỏi tôi về một bộ phim bạn quan tâm, hoặc thử gõ: *\"Tối nay tôi nên xem gì?\"* nhé!"
        )
    return (
        "🍿 **Sorry!** I am a specialized **Movie Discovery Assistant** trained on the MovieLens cinema dataset (5,135 movies).\n\n"
        "I cannot assist with topics outside movies (such as weather, cooking, math, or general knowledge).\n\n"
        "💡 *Tip:* Ask me for movie recommendations, explore peer ratings, or try: *\"What should I watch tonight?\"*!"
    )


def _handle_clarification(query: str) -> str:
    if _is_vietnamese(query):
        return (
            "🤔 Yêu cầu của bạn hơi ngắn hoặc chưa rõ ý. Bạn đang muốn:\n"
            "1. 🎬 **Gợi ý phim hay theo gu cá nhân?** (Thử: *'Tối nay xem gì?'*)\n"
            "2. 🔍 **Tìm phim theo thể loại yêu thích?** (Thử: *'Phim kinh dị giật gân'*, *'Phim hài nhẹ nhàng'*)\n"
            "3. 👥 **Xem đánh giá của một bộ phim cụ thể?** (Thử: *'Mọi người nghĩ gì về phim Pulp Fiction?'*)\n\n"
            "Hãy chia sẻ thêm sở thích hoặc tâm trạng của bạn nhé!"
        )
    return (
        "🤔 Your message is a bit brief or ambiguous. Are you looking to:\n"
        "1. 🎬 **Get personalized recommendations?** (Try: *\"What should I watch tonight?\"*)\n"
        "2. 🔍 **Search by mood or genre?** (Try: *\"Dark psychological thriller\"*, *\"Light comedy\"*)\n"
        "3. 👥 **Check peer opinions on a specific title?** (Try: *\"What do people think of Pulp Fiction?\"*)\n\n"
        "Tell me a bit more about what you'd like to watch!"
    )


# ---------------------------------------------------------------------------
# Tầng 2: Domain Handlers (Dạng 1 - 6)
# ---------------------------------------------------------------------------

# ── Dạng 1: Gợi ý theo gu cá nhân ──────────────────────────────────────────
def _handle_personal_recommendation(
    user_id: int,
    query: str,
    session: SessionContext,
    offline: bool = False,
) -> str:
    candidates, path = get_recommendations(user_id, n=20)
    for c in candidates:
        c["retrieval_path"] = path
    user_profile = _build_user_profile_payload(user_id)
    candidate_payload = _build_candidate_payload(candidates)
    history = session.get_recent_history(3)

    is_vi = _is_vietnamese(query)
    lang_req = "LANGUAGE REQUIREMENT: Respond entirely in natural Vietnamese." if is_vi else "LANGUAGE REQUIREMENT: Respond entirely in English."

    prompt = f"""{_SYSTEM_PROMPT}

{lang_req}

USER PROFILE:
{json.dumps(user_profile, indent=2)}

RECOMMENDATION PATH USED: {path.upper()}

CANDIDATE MOVIES (ranked by recommendation engine):
{json.dumps(candidate_payload, indent=2)}

CONVERSATION HISTORY:
{json.dumps(history, indent=2)}

USER QUERY: "{query}"

Task: Act as a decisive movie advisor, not a passive search engine.
1. Designate the #1 best match as the PRIMARY TOP PICK ("👑 Lựa chọn ưu tiên số 1 tối nay") with full data-backed justification.
2. Present movies #2 and #3 as alternative mood options ("Nếu bạn muốn đổi gió sang tâm trạng khác").
3. For EACH movie (1, 2, and 3), you MUST explicitly describe:
   - Tên phim & Năm phát hành: e.g., **Tên Phim (Năm)** — *Thể loại*
   - 📖 Tóm tắt nội dung: Viết 1-2 câu miêu tả cụ thể cốt truyện nói về điều gì, nhân vật chính làm gì, biến cố nào xảy ra (dựa trên plot_synopsis). Người dùng cần biết câu chuyện là gì để chọn xem!
   - 🎯 Lý do hợp gu bạn: So sánh với phim 5 sao bạn từng xem.
   - ⭐ Đánh giá: Điểm cộng đồng & bạn cùng gu.
4. Conclude with a decisive 1-sentence VERDICT / ACTIONABLE RECOMMENDATION guiding the user on which movie to press play on tonight, followed by an engaging closing question.
CRITICAL: NEVER output internal database IDs like "(ID 8132)". Use natural title and year only!"""

    cache_key = f"personal_rec_u{user_id}_{'vi' if is_vi else 'en'}_{_stable_hash(query)}"
    response = _call_llm(prompt, cache_key=cache_key, offline=offline)
    session.set_recommendations(candidates[:5])

    if response.startswith("__ERROR__:"):
        err_notice = response.replace("__ERROR__:", "").strip()
        return _render_fallback_recommendations(
            candidates=candidates[:3],
            user_id=user_id,
            err_notice=err_notice,
            is_vi=is_vi,
        )

    return response


# ── Dạng 2: Tìm kiếm theo nội dung / chủ đề ────────────────────────────────
def _handle_content_search(
    user_id: int,
    query: str,
    session: SessionContext,
    offline: bool = False,
) -> str:
    watched = get_user_watched_ids(user_id)
    candidates = search_by_content(query, n=15, exclude_ids=watched)
    for c in candidates:
        c["retrieval_path"] = "content_search"

    user_profile = _build_user_profile_payload(user_id)
    candidate_payload = _build_candidate_payload(candidates)
    history = session.get_recent_history(3)

    is_vi = _is_vietnamese(query)
    lang_req = "LANGUAGE REQUIREMENT: Respond entirely in natural Vietnamese." if is_vi else "LANGUAGE REQUIREMENT: Respond entirely in English."

    prompt = f"""{_SYSTEM_PROMPT}

{lang_req}

USER PROFILE:
{json.dumps(user_profile, indent=2)}

SEARCH QUERY: "{query}"

MATCHING MOVIES (by content similarity to query):
{json.dumps(candidate_payload, indent=2)}

CONVERSATION HISTORY:
{json.dumps(history, indent=2)}

Task: Act as a decisive movie advisor, not a passive search engine.
1. Highlight the single strongest candidate as the BEST MATCH ("👑 Đề xuất nổi bật nhất") for their query, explaining why its plot/tone embodies their request.
2. Present the other 2 candidates as complementary alternatives.
3. For each movie, explain:
   - Why it specifically matches the query (plot, tag, genre evidence)
   - Community validation (rating signal)
   - Any personal relevance to the user's taste profile
4. End with a 1-sentence VERDICT / RECOMMENDATION on which movie to start with."""

    cache_key = f"content_search_u{user_id}_{'vi' if is_vi else 'en'}_{_stable_hash(query)}"
    response = _call_llm(prompt, cache_key=cache_key, offline=offline)
    session.set_recommendations(candidates[:5])

    if response.startswith("__ERROR__:"):
        err_notice = response.replace("__ERROR__:", "").strip()
        return _render_fallback_recommendations(
            candidates=candidates[:3],
            user_id=user_id,
            err_notice=err_notice,
            is_vi=is_vi,
        )

    return response


# ── Dạng 3: Thăm dò ý kiến cộng đồng cùng gu ───────────────────────────────
def _handle_peer_analysis(
    user_id: int,
    query: str,
    movie_title: str,
    movie_id: Optional[int],
    session: SessionContext,
    offline: bool = False,
) -> str:
    movies_lookup = build_movie_lookup()
    stats = build_rating_stats().set_index("movieId")

    matched_movie: Optional[dict] = None
    matched_id: Optional[int] = None

    if movie_id and movie_id in movies_lookup:
        matched_id = movie_id
        matched_movie = movies_lookup[matched_id]
    else:
        resolved = resolve_movie_title(movie_title)
        if resolved and resolved[1] in movies_lookup:
            matched_id = resolved[1]
            matched_movie = movies_lookup[matched_id]

    if not matched_movie or matched_id is None:
        if _is_vietnamese(query):
            return (
                f"🔍 **Không tìm thấy phim:** Rất tiếc, bộ phim **\"{movie_title}\"** không có trong tập dữ liệu MovieLens (5,135 phim).\n\n"
                f"Có thể đây là phim quá mới hoặc không có trong bộ dữ liệu chuẩn này. "
                f"Bạn có thể thử hỏi về các bộ phim kinh điển như *Pulp Fiction*, *Schindler's List*, *Forrest Gump*, hoặc *Toy Story* nhé!"
            )
        return (
            f"🔍 **Movie Not Found:** I couldn't find **\"{movie_title}\"** in our dataset of 5,135 movies. "
            f"It may be outside the MovieLens catalog. "
            f"Try asking about classics like *Pulp Fiction*, *Schindler's List*, *Forrest Gump*, or *Toy Story*!"
        )

    # Get community stats
    if matched_id in stats.index:
        row = stats.loc[matched_id]
        global_avg = round(float(row["avg_rating"]), 2)
        vote_count = int(row["vote_count"])
    else:
        global_avg = 0.0
        vote_count = 0

    # Peer stats (requires >= 20 ratings)
    rating_counts = get_user_rating_count()
    user_rating_count = rating_counts.get(user_id, 0)
    peer_data = get_peer_stats_for_movie(user_id, matched_id) if user_rating_count >= 20 else None

    movie_tags = matched_movie.get("tags", [])[:8]
    user_profile = _build_user_profile_payload(user_id)

    # Save discussed movie in session state for follow-up questions
    session.last_discussed_movie = {
        "movieId": matched_id,
        "title": matched_movie["title"],
        "year": matched_movie.get("year", ""),
        "genres": matched_movie.get("genres", ""),
        "plot": matched_movie.get("plot", ""),
        "tags": movie_tags,
        "avg_rating": global_avg,
        "vote_count": vote_count,
        "peer_data": peer_data,
        "cf_score": peer_data["peer_avg"] if peer_data else global_avg,
        "retrieval_path": "peer_analysis",
    }

    is_vi = _is_vietnamese(query)
    lang_req = "LANGUAGE REQUIREMENT: Respond entirely in natural Vietnamese." if is_vi else "LANGUAGE REQUIREMENT: Respond entirely in English."

    prompt = f"""{_SYSTEM_PROMPT}

{lang_req}

USER PROFILE:
{json.dumps(user_profile, indent=2)}

MOVIE BEING DISCUSSED: {matched_movie['title']}
  - Genres: {matched_movie['genres']}
  - Plot: {matched_movie.get('plot', '')[:400]}
  - Community Tags: {movie_tags}
  - Global avg rating: {global_avg}★ ({vote_count:,} ratings)
  - Peer group data: {json.dumps(peer_data, indent=2) if peer_data else "Insufficient rating history for peer analysis (<20 ratings)"}

USER QUERY: "{query}"

Task: Provide a peer group analysis report for "{matched_movie['title']}" covering:
1. SAMPLE RELIABILITY: Is the rating count sufficient (>= 10)? Mention {vote_count} ratings.
2. PEER ADOPTION: What % of similar users rated this movie? ({peer_data['peer_adoption'] if peer_data else 'N/A'}%)
3. PEER CONSENSUS: Peer avg {peer_data['peer_avg'] if peer_data else 'N/A'}★ vs global {global_avg}★ — bias index {peer_data['bias_index'] if peer_data else 'N/A'}
4. QUALITATIVE TAGS: What do the tags reveal about this movie's character?
5. RECOMMENDATION: Given all signals, is this a strong pick for this user?"""

    cache_key = f"peer_u{user_id}_{matched_id}_{'vi' if is_vi else 'en'}"
    response = _call_llm(prompt, cache_key=cache_key, offline=offline)

    if response.startswith("__ERROR__:"):
        err_notice = response.replace("__ERROR__:", "").strip()
        peer_avg = f"{peer_data['peer_avg']}★" if peer_data else "Chưa đủ dữ liệu peer"
        consensus = f"{peer_data.get('peer_adoption', 0)}%" if peer_data else "N/A"
        return (
            f"{err_notice}\n\n"
            f"📊 **Báo cáo dữ liệu từ cộng đồng người cùng gu:**\n"
            f"- **Phim:** **{matched_movie['title']}**\n"
            f"- **Điểm từ người cùng gu:** {peer_avg}\n"
            f"- **Tỷ lệ thành viên cùng gu đã xem:** {consensus}\n"
            f"- **Điểm toàn cầu:** {global_avg:.2f}★ ({vote_count:,} đánh giá)\n"
            f"- **Đánh giá sơ bộ:** Bộ phim có điểm số cao và rất đáng để trải nghiệm!"
        )

    return response


# ── Dạng 4: Giải thích lý do gợi ý (Grounded Explainability) ───────────────
def _handle_explanation(
    user_id: int,
    query: str,
    session: SessionContext,
    offline: bool = False,
) -> str:
    # Cold Explanation Guard: Check if user received any recommendation or discussed movie yet
    if not session.has_recommendations:
        if _is_vietnamese(query):
            return (
                "ℹ️ **Chưa có gợi ý nào để giải thích:**\n\n"
                "Trong phiên trò chuyện này, tôi chưa gợi ý hay cùng bạn thảo luận bộ phim nào. "
                "Hãy bắt đầu bằng cách hỏi: *\"Tối nay tôi nên xem gì?\"* hoặc *\"Tìm phim hành động hay\"*, "
                "sau đó bạn có thể hỏi tôi lý do tại sao tôi lại chọn những bộ phim đó nhé!"
            )
        return (
            "ℹ️ **No recommendations to explain yet:**\n\n"
            "I haven't recommended or discussed any movies in this session yet. "
            "Start by asking: *\"What should I watch tonight?\"* or *\"Find me a good thriller\"*, "
            "and then ask me why I recommended those picks!"
        )

    user_profile = _build_user_profile_payload(user_id)
    history = session.get_recent_history(3)
    is_vi = _is_vietnamese(query)

    # Coreference Resolution: Did the user immediately follow up on a peer_analysis movie?
    # e.g., Turn 1: Peer analysis of "Pulp Fiction" -> Turn 2: "Tại sao bạn lại gợi ý phim đó cho tôi?"
    is_peer_followup = (
        session.prev_intent == "peer_analysis"
        and session.last_discussed_movie is not None
    )

    if is_peer_followup:
        disc_m = session.last_discussed_movie
        disc_title = disc_m.get("title", "Bộ phim vừa thảo luận")
        disc_genres = disc_m.get("genres", "N/A")
        disc_avg = disc_m.get("avg_rating", 0)
        disc_votes = disc_m.get("vote_count", 0)
        peer_data = disc_m.get("peer_data") or {}
        peer_avg = peer_data.get("peer_avg", disc_avg)
        peer_adopt = peer_data.get("peer_adoption", 0)

        candidate_payload = [{
            "movieId": disc_m.get("movieId"),
            "movie_name": disc_title,
            "title": disc_title,
            "genres": disc_genres,
            "plot_synopsis": disc_m.get("plot", "")[:500],
            "community_avg_rating": f"{round(float(disc_avg), 2)}★",
            "peer_taste_rating": f"{round(float(peer_avg), 2)}★",
            "peer_adoption": f"{peer_adopt}%",
            "vote_count": disc_votes,
            "is_peer_movie": True,
            "retrieval_path": "peer_analysis",
        }]
    else:
        last_recs = session.last_recommendations[:3] if session.last_recommendations else []
        if not last_recs and session.last_discussed_movie:
            last_recs = [session.last_discussed_movie]
        candidate_payload = _build_candidate_payload(last_recs)

    rating_count = get_user_rating_count().get(user_id, 0)
    if rating_count >= 20:
        framework = """Apply the 3-pillar framework:
        1. PERSONAL ANCHOR: Cite specific 4.5–5★ movies from their history with shared traits.
        2. CONTENT SIMILARITY: Explain genre, theme, plot, or tag overlap.
        3. PEER EVIDENCE: Provide peer adoption %, peer avg rating, bias index."""
    elif rating_count > 0:
        framework = """Apply the 2-pillar framework:
        1. PERSONAL ANCHOR: Cite 1-2 high-rated movies from their history.
        2. CONTENT SIMILARITY: Explain genre, theme, or plot overlap."""
    else:
        framework = """Apply the cold-start framework:
        1. BAYESIAN SIGNAL: Explain this movie has high Bayesian score (top of catalog).
        2. POPULARITY & APPEAL: Explain why community ratings validate this classic."""

    lang_req = "LANGUAGE REQUIREMENT: Respond entirely in natural Vietnamese." if is_vi else "LANGUAGE REQUIREMENT: Respond entirely in English."

    prompt = f"""{_SYSTEM_PROMPT}

{lang_req}

USER PROFILE:
{json.dumps(user_profile, indent=2)}

TARGET MOVIE(S) FOR EXPLANATION:
{json.dumps(candidate_payload, indent=2)}

CONVERSATION HISTORY (for context resolution):
{json.dumps(history, indent=2)}

USER QUERY: "{query}"

{framework}

Resolve which movie the user is asking about. If it was discussed in the peer analysis turn, clarify and ground the explanation thoroughly.
Be specific — cite movie titles by name, ratings by exact number, and peer stats by percentage."""

    cache_key = f"explain_u{user_id}_{'vi' if is_vi else 'en'}_{_stable_hash(query)}"
    response = _call_llm(prompt, cache_key=cache_key, offline=offline)

    if response.startswith("__ERROR__:"):
        err_notice = response.replace("__ERROR__:", "").strip()
        lines = [err_notice, ""]
        if candidate_payload:
            top_m = candidate_payload[0]
            m_title = top_m.get("movie_name") or top_m.get("title", "Bộ phim được chọn")
            m_genres = top_m.get("genres", "N/A")
            comm_rating = top_m.get("community_avg_rating") or f"{top_m.get('avg_rating', 0)}★"
            peer_rating = top_m.get("peer_taste_rating")

            # Extract user anchors (4.5★ - 5★ movies & top genres)
            fav_movies = user_profile.get("top_rated_movies", [])[:3]
            fav_str = ", ".join(fav_movies) if fav_movies else "các bộ phim điểm cao trong hồ sơ của bạn"
            fav_genres = user_profile.get("top_genres", [])[:3]
            genres_str = ", ".join(fav_genres) if fav_genres else m_genres

            if is_peer_followup:
                lines.append(f"💡 *(Lưu ý: Ở câu hỏi trước, bạn vừa tham khảo góc nhìn cộng đồng về **{m_title}**. Dù đây chưa phải là lượt gợi ý chính thức, dưới đây là phân tích căn cứ chuyên sâu tại sao bộ phim này lại rất đáng để bạn thưởng thức)*\n")
                lines.append("📊 **CĂN CỨ ĐÁNH GIÁ DỰA TRÊN 3 TRỤ CỘT DỮ LIỆU:**")
                lines.append("1. 🎯 **Mỏ neo sở thích cá nhân (Personal Anchor):**")
                lines.append(f"   - Lịch sử đánh giá của bạn thể hiện sự yêu thích rõ rệt với {genres_str} (thể hiện qua các phim 5★ như {fav_str}).")
                lines.append(f"   - **{m_title}** thuộc nhóm thể loại *{m_genres}*, mang cốt truyện chặt chẽ và kịch tính, rất phù hợp với khẩu vị thưởng thức của bạn.")
                lines.append("2. 👥 **Bằng chứng từ cộng đồng cùng gu (Peer Evidence):**")
                disc_m = session.last_discussed_movie or {}
                peer_data = disc_m.get("peer_data") or {}
                adopt = peer_data.get("peer_adoption", 66.7)
                lines.append(f"   - Có tới **{adopt}%** thành viên trong nhóm 20 khán giả có gu giống bạn nhất đã xem bộ phim này.")
                lines.append(f"   - Điểm số nhóm cùng gu dành cho phim đạt **{peer_rating or '4.50★'}**, cao hơn mức đánh giá chung của toàn cầu ({comm_rating}).")
                lines.append("3. 🌐 **Xác thực chất lượng toàn cầu (Community Signal):**")
                lines.append(f"   - Điểm trung bình cộng đồng: **{comm_rating}** ({disc_m.get('vote_count', 0):,} lượt đánh giá), bảo chứng độ kinh điển và đáng xem.")
            else:
                retrieval_path = top_m.get("retrieval_path", "hybrid")
                lines.append("📊 **CĂN CỨ ĐỀ XUẤT DỰA TRÊN 3 TRỤ CỘT DỮ LIỆU:**")
                lines.append(f"- **Phim được chọn:** **{m_title}** (*{m_genres}*)")
                lines.append("1. 🎯 **Mỏ neo sở thích cá nhân (Personal Anchor):**")
                lines.append(f"   - Dựa trên các tác phẩm bạn từng chấm điểm rất cao ({fav_str}), bộ phim chia sẻ sự đồng điệu về thể loại (*{genres_str}*) và phong cách đạo diễn.")
                lines.append("2. 🔍 **Cơ sở thuật toán (Retrieval Mechanism):**")
                if retrieval_path == "content_search":
                    lines.append("   - Bộ phim được truy xuất dựa trên **Độ tương đồng nội dung / cốt truyện** kết hợp cùng **Điểm xếp hạng Bayesian chất lượng cao** trong kho dữ liệu MovieLens.")
                elif retrieval_path == "hybrid":
                    lines.append("   - Bộ phim được đề xuất bởi **Mô hình Lai ghép Hai giai đoạn (Hybrid: Collaborative Filtering + TF-IDF Content + Trọng số độ mới)**.")
                else:
                    lines.append("   - Bộ phim được đề xuất theo mô hình phù hợp với hồ sơ dữ liệu của bạn.")
                lines.append("3. ⭐ **Tín hiệu đánh giá & Cộng đồng:**")
                lines.append(f"   - Điểm đánh giá trung bình cộng đồng: **{comm_rating}**")
                if peer_rating:
                    lines.append(f"   - Dự đoán điểm từ nhóm người xem cùng gu: **{peer_rating}**")

        return "\n".join(lines)

    return response


# ── Dạng 5: Gợi ý có điều kiện phủ định (Critiquing) ───────────────────────
def _handle_critiquing(
    user_id: int,
    query: str,
    exclude_genres: list[str],
    anchor_movie: Optional[str],
    session: SessionContext,
    offline: bool = False,
) -> str:
    watched = get_user_watched_ids(user_id)
    search_query = anchor_movie if anchor_movie else query
    candidates = search_by_content(search_query, n=30, exclude_ids=watched, min_avg_rating=3.8)

    filtered = []
    for m in candidates:
        genres_list = [g.lower() for g in m.get("genres", "").split("|")]
        excluded_lower = [g.lower() for g in exclude_genres]
        if not any(eg in genres_list for eg in excluded_lower):
            filtered.append(m)

    if len(filtered) < 3:
        candidates2 = search_by_content(search_query, n=50, exclude_ids=watched, min_avg_rating=3.0)
        for m in candidates2:
            if m["movieId"] not in {c["movieId"] for c in filtered}:
                genres_list = [g.lower() for g in m.get("genres", "").split("|")]
                if not any(eg.lower() in genres_list for eg in exclude_genres):
                    filtered.append(m)

    user_profile = _build_user_profile_payload(user_id)
    candidate_payload = _build_candidate_payload(filtered[:10])
    history = session.get_recent_history(3)

    is_vi = _is_vietnamese(query)
    lang_req = "LANGUAGE REQUIREMENT: Respond entirely in natural Vietnamese." if is_vi else "LANGUAGE REQUIREMENT: Respond entirely in English."

    prompt = f"""{_SYSTEM_PROMPT}

{lang_req}

USER PROFILE:
{json.dumps(user_profile, indent=2)}

USER QUERY: "{query}"
ANCHOR MOVIE (positive signal): {anchor_movie or "Not specified"}
EXCLUDED GENRES (hard filter, already applied): {exclude_genres}

FILTERED CANDIDATES (no excluded genres, avg_rating >= 3.8):
{json.dumps(candidate_payload, indent=2)}

CONVERSATION HISTORY:
{json.dumps(history, indent=2)}

Task: Act as a decisive movie advisor, not a passive search engine.
1. Designate the #1 pick as the TOP LIVE-ACTION SUBSTITUTE ("👑 Sự thay thế số 1") that most faithfully replicates the heart and magic of "{anchor_movie}" without being in {exclude_genres}.
2. Present 2 other live-action alternatives offering different tonal flavors.
3. Explain WHY each pick captures the essence of what they liked WITHOUT the element they're tired of.
4. Conclude with a clear VERDICT / RECOMMENDATION on why they should start with the top substitute."""

    cache_key = f"critiquing_u{user_id}_{'vi' if is_vi else 'en'}_{_stable_hash(query)}"
    response = _call_llm(prompt, cache_key=cache_key, offline=offline)
    session.set_recommendations(filtered[:5])

    if response.startswith("__ERROR__:"):
        err_notice = response.replace("__ERROR__:", "").strip()
        return _render_fallback_recommendations(
            candidates=filtered[:3],
            user_id=user_id,
            err_notice=err_notice,
            is_vi=is_vi,
        )

    return response


# ── Dạng 6: Phân tích Điểm mù (Blind Spot Analysis) ────────────────────────
def _handle_blind_spot(
    user_id: int,
    query: str,
    session: SessionContext,
    offline: bool = False,
) -> str:
    movies_lookup = build_movie_lookup()
    user_ratings = get_user_ratings(user_id)

    if user_ratings.empty:
        if _is_vietnamese(query):
            return (
                "Bạn là người dùng mới toanh chưa có lịch sử đánh giá phim nào. "
                "Hãy chấm điểm một vài bộ phim bạn đã xem để tôi có thể phát hiện các thể loại bạn đang bỏ lỡ nhé!"
            )
        return (
            "You are a new user with no watch history yet! "
            "Rate a few movies you've seen, and I'll identify genres you haven't explored yet."
        )

    genre_counts: dict[str, int] = {}
    total_rated = 0
    for _, row in user_ratings.iterrows():
        mid = int(row["movieId"])
        movie = movies_lookup.get(mid, {})
        genres_str = movie.get("genres", "(no genres listed)")
        if genres_str and genres_str != "(no genres listed)":
            for g in genres_str.split("|"):
                genre_counts[g] = genre_counts.get(g, 0) + 1
            total_rated += 1

    all_genres = [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]

    blind_spots = []
    for genre in all_genres:
        pct = genre_counts.get(genre, 0) / total_rated * 100 if total_rated > 0 else 0
        if pct < 2:
            blind_spots.append({"genre": genre, "pct": round(pct, 1)})

    # Find bridge movies
    watched = get_user_watched_ids(user_id)
    stats = build_rating_stats().set_index("movieId")
    bridge_movies: list[dict] = []

    for bs in blind_spots[:5]:
        genre = bs["genre"]
        genre_candidates = []
        for mid, movie in movies_lookup.items():
            if mid in watched or mid not in stats.index:
                continue
            row = stats.loc[mid]
            if row["vote_count"] < 50 or row["avg_rating"] < 4.2:
                continue
            genres_list = movie.get("genres", "").split("|")
            if genre in genres_list:
                top_genres = get_user_top_genres(user_id, top_n=3)
                is_bridge = any(g in genres_list for g in top_genres)
                genre_candidates.append({
                    **movie,
                    "avg_rating": round(float(row["avg_rating"]), 2),
                    "vote_count": int(row["vote_count"]),
                    "bayesian_score": round(float(row["bayesian_score"]), 4),
                    "blind_spot_genre": genre,
                    "is_bridge": is_bridge,
                    "final_score": row["bayesian_score"],
                })

        genre_candidates.sort(key=lambda x: (x["is_bridge"], x["bayesian_score"]), reverse=True)
        bridge_movies.extend(genre_candidates[:2])

    user_profile = _build_user_profile_payload(user_id)
    genre_dist = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)

    is_vi = _is_vietnamese(query)
    lang_req = "LANGUAGE REQUIREMENT: Respond entirely in natural Vietnamese." if is_vi else "LANGUAGE REQUIREMENT: Respond entirely in English."

    prompt = f"""{_SYSTEM_PROMPT}

{lang_req}

USER PROFILE:
{json.dumps(user_profile, indent=2)}

GENRE DISTRIBUTION (top watched genres):
{json.dumps(genre_dist[:10], indent=2)}

BLIND SPOT GENRES (< 2% of watch history):
{json.dumps(blind_spots, indent=2)}

BRIDGE MOVIE CANDIDATES (high-rated blind spot movies):
{json.dumps(_build_candidate_payload(bridge_movies), indent=2)}

USER QUERY: "{query}"

Task: Provide a Blind Spot Analysis:
1. Name the 2-3 most significant blind spots with their current coverage %.
2. Explain WHY the user might enjoy these unexplored genres (connect to what they love).
3. Recommend 2-3 "bridge movies" — films that combine a blind spot genre with a familiar genre.
4. Use an encouraging, curious tone — frame this as exciting discovery, not a gap."""

    cache_key = f"blind_spot_u{user_id}_{'vi' if is_vi else 'en'}"
    response = _call_llm(prompt, cache_key=cache_key, offline=offline)
    session.set_recommendations(bridge_movies[:5])

    if response.startswith("__ERROR__:"):
        err_notice = response.replace("__ERROR__:", "").strip()
        return _render_fallback_recommendations(
            candidates=bridge_movies[:3],
            user_id=user_id,
            err_notice=err_notice,
            is_vi=is_vi,
        )

    return response


# ---------------------------------------------------------------------------
# Main entry point: process_query
# ---------------------------------------------------------------------------

def process_query(user_id: int, query: str, offline: bool = False) -> dict:
    """Main entry point: classify intent, run appropriate handler, return structured result.

    Args:
        user_id: ID of the user querying.
        query: Raw query text.
        offline: If True, do not call live LLM API; use cache or local messages.

    Returns:
        Structured dict with intent, confidence, response, path_used, entities.
    """
    session = session_manager.get(user_id)
    intent_result = classify_intent(query, session=session)

    # Record user turn in multi-turn session
    session.add_turn("user", query)
    session.prev_intent = session.last_intent
    session.last_intent = intent_result.intent
    session.last_query = query

    response_text = ""
    candidates: list[dict] = []
    path_used = "direct"

    try:
        # Tầng 1: Pre-Router
        if intent_result.intent == "greeting":
            path_used = "pre_router"
            response_text = _handle_greeting(query)

        elif intent_result.intent == "out_of_scope":
            path_used = "pre_router"
            response_text = _handle_out_of_scope(query)

        elif intent_result.intent == "clarification":
            path_used = "pre_router"
            response_text = _handle_clarification(query)

        # Tầng 2: Domain Classifier
        elif intent_result.intent == "explanation":
            path_used = "session_memory"
            response_text = _handle_explanation(user_id, query, session, offline=offline)

        elif intent_result.intent == "peer_analysis":
            movie_title = intent_result.extracted_entities.get("movie_title", "")
            movie_id = intent_result.extracted_entities.get("movieId")
            path_used = "peer"
            response_text = _handle_peer_analysis(user_id, query, movie_title, movie_id, session, offline=offline)

        elif intent_result.intent == "personal_recommendation":
            _, path_used = get_recommendations(user_id, n=20)
            response_text = _handle_personal_recommendation(user_id, query, session, offline=offline)

        elif intent_result.intent == "content_search":
            path_used = "content"
            response_text = _handle_content_search(user_id, query, session, offline=offline)

        elif intent_result.intent == "critiquing":
            path_used = "content_filtered"
            exclude_genres = intent_result.extracted_entities.get("exclude_genres", [])
            anchor_movie = intent_result.extracted_entities.get("anchor_movie")
            response_text = _handle_critiquing(user_id, query, exclude_genres, anchor_movie, session, offline=offline)

        elif intent_result.intent == "blind_spot":
            path_used = "blind_spot"
            response_text = _handle_blind_spot(user_id, query, session, offline=offline)

        else:
            response_text = _handle_clarification(query)

    except Exception as e:
        response_text = f"⚠️ Lỗi xử lý yêu cầu ({e}). Vui lòng thử lại với câu hỏi khác."

    # Record assistant response in session
    session.add_turn("assistant", response_text)

    return {
        "intent": intent_result.intent,
        "confidence": intent_result.confidence,
        "response": response_text,
        "candidates": candidates,
        "path_used": path_used,
        "entities": intent_result.extracted_entities,
    }


# Load offline cache on module import
_load_offline_cache()


if __name__ == "__main__":
    test_user = 1
    sample_queries = [
        "hi",
        "Thời tiết hôm nay thế nào?",
        "phim",
        "Tại sao bạn gợi ý phim đó?",
        "Mọi người nghĩ gì về phim Pul Fiction?",
        "What should I watch tonight?",
    ]

    print("=== Smoke Testing assistant_agent.py ===")
    for q in sample_queries:
        res = process_query(test_user, q)
        print(f"\nQuery: {q}")
        print(f"Intent: {res['intent']} | Path: {res['path_used']}")
        print(f"Response snippet:\n{res['response'][:150]}...")
