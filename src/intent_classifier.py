"""
intent_classifier.py — High-Precision Three-tier Intent Classification with Guardrails.

Tầng 1: Pre-Router & Normalizer (< 1ms, Free)
    - Security / Jailbreak / Injection Detection (SQLi, XSS, Prompt Injection -> out_of_scope)
    - Greeting, Courtesy, Gratitude & Farewell (hi, xin chào, cảm ơn, tạm biệt -> greeting)
    - Out-of-Scope (weather, cooking, math, civics, general IT -> out_of_scope)
    - Clarification (terse, vague, emotional queries like "phim", "chán quá" -> clarification)

Tầng 2: Enhanced Domain Classifier + High-Precision Fuzzy Matching (< 5ms)
    - peer_analysis: Title extraction with alias expansion & tuned difflib fuzzy matching
    - explanation: Grounded reasoning with cold explanation guard
    - blind_spot: Genre frequency under-exploration
    - critiquing: Anchor movie + hard exclusion filtering
    - content_search: Keyword / plot / mood TF-IDF search
    - personal_recommendation: Catch-all personalized recommendation
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from session_manager import SessionContext

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    intent: str                      # greeting | out_of_scope | clarification | 6 domain intents
    confidence: str                  # "high" | "medium" | "low"
    extracted_entities: dict         # e.g., {"movie_title": "Pulp Fiction", "exclude_genres": ["Animation"]}
    raw_query: str

    def __repr__(self) -> str:
        return f"IntentResult(intent={self.intent!r}, confidence={self.confidence!r}, entities={self.extracted_entities})"


# ---------------------------------------------------------------------------
# Vietnamese Normalization Helper
# ---------------------------------------------------------------------------

def strip_accents(s: str) -> str:
    """Strip Vietnamese diacritics and convert to lowercase for robust pattern matching."""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()


# ---------------------------------------------------------------------------
# High-Precision Fuzzy Title Resolution (difflib with Guards & Aliases)
# ---------------------------------------------------------------------------

_title_lookup: Optional[dict[str, tuple[str, int]]] = None

def _get_title_lookup() -> dict[str, tuple[str, int]]:
    """Build lookup map with alias expansion for MovieLens title conventions."""
    global _title_lookup
    if _title_lookup is None:
        try:
            from data_loader import load_movies
            movies = load_movies()
            lookup: dict[str, tuple[str, int]] = {}
            for _, row in movies.iterrows():
                raw_title = str(row["title"])
                mid = int(row["movieId"])
                # Strip trailing year like (1994)
                clean = re.sub(r"\s*\(\d{4}\)$", "", raw_title).strip().lower()

                variations = [clean, strip_accents(clean)]

                # Handle MovieLens trailing articles: 'Godfather, The' -> 'the godfather' & 'godfather'
                m_the = re.match(r"^(.*?),\s*(the|a|an)$", clean)
                if m_the:
                    base, art = m_the.group(1), m_the.group(2)
                    variations.extend([
                        f"{art} {base}",
                        base,
                        strip_accents(f"{art} {base}"),
                        strip_accents(base),
                    ])

                # Handle subtitles: 'Terminator 2: Judgment Day' -> 'terminator 2'
                if ":" in clean:
                    prefix = clean.split(":")[0].strip()
                    variations.extend([prefix, strip_accents(prefix)])

                # Handle aliases: 'Seven (a.k.a. Se7en)' -> 'seven', 'se7en'
                if " (a.k.a. " in clean:
                    main_t = clean.split(" (a.k.a. ")[0].strip()
                    aka = clean.split(" (a.k.a. ")[1].rstrip(")").strip()
                    variations.extend([main_t, aka, strip_accents(main_t), strip_accents(aka)])

                for v in variations:
                    if v:
                        lookup[v] = (raw_title, mid)

            _title_lookup = lookup
        except Exception:
            _title_lookup = {}
    return _title_lookup


def resolve_movie_title(query_title: str, cutoff: float = 0.78) -> Optional[tuple[str, int]]:
    """Fuzzy match a query title against catalog with strict length & word guardrails.
    
    Prevents false over-matching (e.g. 'The Matrix' matching 'The Artist', or 'Barbie' matching 'Babe').
    """
    if not query_title:
        return None
    lookup = _get_title_lookup()
    if not lookup:
        return None

    q_clean = query_title.strip().lower()
    # Strip common leading and trailing noise words
    q_clean = re.sub(r"^(?:phim|movie|film|the)\s+", "", q_clean).strip()
    q_clean = re.sub(r"\s+(?:phim|movie|film|hoạt hình|hoat hinh)$", "", q_clean).strip()
    q_no_accents = strip_accents(q_clean)

    # 1. Exact match checks
    if q_clean in lookup:
        return lookup[q_clean]
    if q_no_accents in lookup:
        return lookup[q_no_accents]
    if f"the {q_clean}" in lookup:
        return lookup[f"the {q_clean}"]

    # 2. Strict fuzzy search
    candidates = list(lookup.keys())
    matches = difflib.get_close_matches(q_clean, candidates, n=3, cutoff=cutoff)
    if not matches:
        matches = difflib.get_close_matches(q_no_accents, candidates, n=3, cutoff=cutoff)

    for match in matches:
        # Guard 1: Short word absolute length difference check
        if len(q_clean) <= 7 and abs(len(q_clean) - len(match)) > 1:
            continue

        # Guard 2: General length ratio check
        sim_ratio = difflib.SequenceMatcher(None, q_clean, match).ratio()
        min_len_ratio = 0.65 if sim_ratio >= 0.82 else 0.72
        len_ratio = min(len(q_clean), len(match)) / max(len(q_clean), len(match))
        if len_ratio < min_len_ratio:
            continue

        # Guard 2: Single-word query should not match multi-word titles unless first word matches
        words_q = q_clean.split()
        words_m = match.split()
        if len(words_q) == 1 and len(words_m) > 1:
            first_word_sim = difflib.SequenceMatcher(None, words_q[0], words_m[0]).ratio()
            if first_word_sim < 0.88:
                continue

        return lookup[match]

    return None


# ---------------------------------------------------------------------------
# Tầng 1: Security & Jailbreak Guardrail
# ---------------------------------------------------------------------------

def _is_security_or_jailbreak(q: str) -> bool:
    """Detect prompt injection, SQL injection, XSS, and dangerous out-of-scope commands."""
    q_norm = strip_accents(q.strip().lower())
    security_patterns = [
        # SQL Injection
        r"\b(drop\s+table|select\s+.+\s+from|insert\s+into|update\s+.+\s+set|delete\s+from|alter\s+table)\b",
        r"'\s*or\s+['\d\w]+\s*=\s*['\d\w]+",
        r"--",
        # Prompt Injection & Jailbreak
        r"\b(ignore\s+(all\s+)?previous|system\s+prompt|system\s+instruction|prompt\s+he\s+thong)\b",
        r"\b(bo\s+qua\s+(tat\s+ca\s+|moi\s+)?huong\s+dan|in\s+ra\s+system\s+prompt)\b",
        r"\b(file\s+\.env|in\s+ra.*\.env)\b",
        # XSS & System Command
        r"<\s*script\b",
        r"(format\s+[a-z]:|shutdown|rm\s+-rf)",
        # Malicious & Harmful Prompts
        r"\b(hack\s+website|che\s+tao\s+vu\s+khi|make\s+a\s+bomb|chui\s+the|insulting\s+poem|tuc\s+tiu)\b",
        r"\b(co\s+phieu.*lam\s+giau|lam\s+giau\s+nhanh.*co\s+phieu)\b",
        r"\b(bao\s+nhieu\s+bang\s+trong\s+database|tables\s+in\s+database)\b",
    ]
    for pattern in security_patterns:
        if re.search(pattern, q_norm, re.IGNORECASE):
            return True
    return False


# ---------------------------------------------------------------------------
# Tầng 1: Pre-Router Functions
# ---------------------------------------------------------------------------

_MOVIE_CORE_TERMS = {
    "phim", "movie", "film", "xem", "watch", "dien vien", "dao dien",
    "the loai", "genre", "recommend", "goi y", "de xuat", "suggest",
    "rap", "cinema", "trailer", "rating", "danh gia",
}

def _has_movie_term(q_norm: str) -> bool:
    """Check if query contains any movie core term using strict word boundaries."""
    return any(re.search(rf"\b{re.escape(term)}\b", q_norm) for term in _MOVIE_CORE_TERMS)

def _is_greeting_or_courtesy(query: str) -> bool:
    """Detect if query is a greeting, pleasantry, gratitude, farewell, or small-talk."""
    q_norm = strip_accents(query.strip().rstrip("?.,!"))
    words = q_norm.split()
    if not words:
        return False

    if _has_movie_term(q_norm):
        return False

    courtesy_exact = {
        # Greetings
        "hi", "hello", "hey", "alo", "helo", "hi bot", "chao", "xin chao", "chao ban", "chao em",
        "good morning", "good evening", "howdy", "howdy assistant", "sup", "sup bot", "greetings",
        "chao buoi sang", "chao buoi toi",
        # Gratitude
        "cam on", "cam on ban", "cam on nhieu", "cam on ban nhieu", "cam on nhe", "cam on rat nhieu",
        "thank you", "thanks", "thanks a lot", "thank you very much", "thank you so much",
        # Farewells
        "tam biet", "tam biet ban", "tam biet nhe", "bye", "bye bye", "goodbye", "hen gap lai", "hen gap lai nhe",
        # Small talk / Wellbeing
        "ban khoe khong", "how are you", "how are you doing", "how are you doing today", "khoe khong",
        # Identity
        "ban ten la gi", "ban la ai", "who are you", "what is your name",
    }
    if q_norm in courtesy_exact:
        return True

    courtesy_prefixes = (
        "xin chao", "chao", "hi", "hello", "hey", "alo", "howdy", "good morning", "good evening",
        "cam on", "thank you", "thanks", "tam biet", "bye", "goodbye",
    )
    if any(q_norm.startswith(p) for p in courtesy_prefixes) and len(words) <= 5:
        return True

    return False


def _is_out_of_scope(query: str) -> bool:
    """Detect queries completely unrelated to cinema/movies or security threats."""
    if _is_security_or_jailbreak(query):
        return True

    q_norm = strip_accents(query.strip().lower())
    if _has_movie_term(q_norm):
        return False

    out_patterns = [
        # Weather
        r"\b(thoi\s+tiet|weather|rain|mua\b|nang\b|nhiet\s+do|temperature)\b",
        # Math & Calculations
        r"\b(giai\s+toan|math|dao\s+ham|phuong\s+trinh|calculate|divided\s+by|chia\s+cho|nhan\s+voi|bang\s+bao\s+nhieu|bang\s+may)\b",
        r"\b\d+\s*[\+\-\*\/]\s*\d+\b",
        r"\b\d+\s*(cong|tru|nhan|chia)\s*\d+\b",
        # Cooking & Food
        r"\b(nau\b|recipe|mon\s+an|cach\s+lam|cook|food|bun\s+bo|pho\s+bo|banh\s+mi|cocktail|flan|cake|soup)\b",
        # Politics, Civics, Geography
        r"\b(tong\s+thong|president|chinh\s+tri|thu\s+do|capital\s+of|dan\s+so|population|world\s+cup|bong\s+da|football|tran\s+dau)\b",
        # IT & Coding outside movie context
        r"\b(viet\s+code|python|scrape|programming|webpage|how\s+to\s+code|deploy.*aws|docker|linux|quicksort|c\+\+|man\s+hinh\s+xanh|windows)\b",
    ]
    for pattern in out_patterns:
        if re.search(pattern, q_norm):
            return True
    return False


def _is_clarification(query: str) -> bool:
    """Detect overly terse, ambiguous or pure emotional queries needing clarification."""
    q_norm = strip_accents(query.strip().rstrip("?.,!"))
    vague_phrases = {
        "phim", "xem", "film", "movie", "watch", "nua", "tiep", "help",
        "cho xem", "goi y", "xem gi", "gi cung duoc", "bat ky", "tuy", "tuy ban",
        "chan qua", "buon qua", "met qua", "chan", "buon",
    }
    return q_norm in vague_phrases or (len(q_norm.split()) <= 2 and q_norm in vague_phrases)


# ---------------------------------------------------------------------------
# Tầng 2: Symmetrical Normalized Keywords
# ---------------------------------------------------------------------------

_EXPLANATION_KEYWORDS = [
    "why", "how come", "reason", "explain", "what makes", "convince me",
    "tell me why", "elaborate", "justify", "choice",
    "tai sao", "vi sao", "ly do", "giai thich", "sao lai", "lam sao biet",
    "thuyet phuc toi", "can cu vao dau",
]

_PEER_PATTERNS = [
    # English
    r"(?:what do|what does|how do|how does).{0,30}(?:similar|like me|my taste|people like me|peers|users).{0,30}think (?:about|of) (.+)",
    r"what (?:do|does) (?:people|others|users|peers|someone).{0,30}(?:think|say|feel) (?:about|of) (.+)",
    r"(?:peer|community|others).{0,30}(?:rating|think|say).{0,30}(?:about|for|of) (.+)",
    r"think (?:about|of) (.+)",
    r"opinion on (.+)",
    r"what about (.+)",
    r"why not (.+)",
    # Vietnamese
    r"(?:mọi người|người dùng|cộng đồng|bạn bè|những người|moi nguoi|nguoi dung|cong dong).{0,30}(?:nghĩ gì|đánh giá|cảm nhận|nghi gi|danh gia|cam nhan).{0,30}(?:về|của|ve|cua) (.+)",
    r"(?:nghĩ gì|cảm nhận|đánh giá|nghi gi|cam nhan|danh gia).{0,30}(?:về|của|ve|cua) (.+)",
    r"(?:mọi người|cộng đồng|người dùng|moi nguoi|cong dong)\s+(?:đánh giá|cảm nhận|danh gia|cam nhan)\s+(?:về\s+)?(?:phim\s+)?(.+?)(?:\s+thế nào|\s+như thế nào|\s+ra sao|\?|$)",
    r"(?:người có gu|người có sở thích|nguoi co gu|nguoi co so thich).{0,30}(?:giống tôi|tương tự|giong toi|tuong tu).{0,30}(?:xem|nghĩ về|thích|nghi ve|thich) (.+)",
    r"ý kiến.{0,20}(?:về|của) (.+)",
    r"y kien.{0,20}(?:ve|cua) (.+)",
    r"phim (.+) có hay không",
    r"phim (.+) co hay khong",
    r"^đánh giá\s+(?:phim\s+)?(.+)",
    r"^danh gia\s+(?:phim\s+)?(.+)",
    r"review phim (.+)",
    r"tại sao không phải là phim (.+)",
    r"tai sao khong phai la phim (.+)",
    r"tại sao không phải là (.+)",
    r"tai sao khong phai la (.+)",
]

_BLIND_SPOT_KEYWORDS = [
    "blind spot", "missing", "genres am i missing", "what genre",
    "haven't explored", "overlooked", "what else should i try", "categories i've overlooked",
    "diem mu", "the loai nao toi chua", "the loai toi bo lo", "bo qua the loai",
    "chua kham pha", "thieu sot", "chua thu the loai", "the loai toi thieu",
    "toi dang bo lo", "toi chua xem the loai",
]

_CRITIQUING_KEYWORDS = [
    "but not", "tired of", "except", "no more", "without", "avoid", "don't want",
    "not in the mood for", "sick of", "bored of", "other than", "not animated", "not horror",
    "nhung khong", "chan phim", "khong muon", "ngoai tru", "tranh", "khong thich nua",
    "khong phai", "nhung khac", "thich nhung", "nhung chan", "nhung ma khong",
]

_CONTENT_SEARCH_KEYWORDS = [
    "i want", "looking for", "something like", "in the mood for", "find me",
    "show me", "recommend something", "suggest", "search for", "similar style", "switch to",
    "dark", "thriller", "comedy", "action", "romantic", "sci-fi", "horror",
    "animated", "documentary", "mystery", "crime", "twist", "worst", "do nhat", "ngan nhat",
    "no genres", "no genres listed", "vua hai", "khoc",
    "toi muon", "tim phim", "phim kinh di", "phim hanh dong", "phim tinh cam",
    "phim hai", "phim hoat hinh", "phim khoa hoc vien tuong", "phim bi an",
    "phim toi pham", "tam ly", "phim tam ly", "phim den toi", "tim cho toi",
    "goi y phim", "phim co twist", "phim co cot truyen", "de xuat phim",
    "muon xem", "tim kiem phim", "phim lang man", "phim vien tuong", "doi sang", "toi tam",
]

_PERSONAL_REC_KEYWORDS = [
    "what should i watch", "recommend", "what to watch", "tonight", "this weekend",
    "what's good", "top movies", "best movies", "suggest a movie", "for me", "more recommendations",
    "other good ones", "other than that", "any more", "what else",
    "toi nay xem gi", "xem gi", "goi y cho toi", "nen xem gi", "phim hay",
    "phim nao hay", "cuoi tuan xem gi", "xem phim gi", "phim tot",
    "bo phim hay", "de xuat cho toi", "phim nao nen xem", "cho toi 3 phim", "phim khac", "con phim nao",
    "cho toi vai bo phim", "phim kinh dien", "vai bo phim",
]


def _extract_movie_title_from_peer_query(query: str) -> Optional[str]:
    """Try to extract the movie title from a peer analysis query."""
    q_norm = strip_accents(query)
    for pattern in _PEER_PATTERNS:
        # Check original
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            extracted = m.group(1).strip().rstrip("?.,!")
            extracted = re.sub(r"^(?:phim|movie)\s+", "", extracted, flags=re.IGNORECASE)
            return extracted.strip()
        # Check normalized
        m_norm = re.search(pattern, q_norm, re.IGNORECASE)
        if m_norm:
            extracted = m_norm.group(1).strip().rstrip("?.,!")
            extracted = re.sub(r"^(?:phim|movie)\s+", "", extracted, flags=re.IGNORECASE)
            return extracted.strip()

    quoted = re.search(r'"([^"]+)"', query)
    if quoted:
        return quoted.group(1).strip()
    return None


def _extract_exclude_genres(query: str) -> list[str]:
    """Extract genres the user wants to avoid from a critiquing query."""
    genre_map = {
        "animated": "Animation", "animation": "Animation", "hoat hinh": "Animation",
        "horror": "Horror", "scary": "Horror", "kinh di": "Horror",
        "comedy": "Comedy", "funny": "Comedy", "hai": "Comedy",
        "action": "Action", "hanh dong": "Action",
        "romantic": "Romance", "romance": "Romance", "tinh cam": "Romance", "lang man": "Romance",
        "documentary": "Documentary", "tai lieu": "Documentary",
        "sci-fi": "Sci-Fi", "science fiction": "Sci-Fi", "khoa hoc vien tuong": "Sci-Fi",
        "fantasy": "Fantasy", "vien tuong": "Fantasy",
        "musical": "Musical", "nhac kich": "Musical",
        "thriller": "Thriller", "drama": "Drama", "mystery": "Mystery", "crime": "Crime",
    }

    q_norm = strip_accents(query)
    excluded = []

    for keyword, genre in genre_map.items():
        if keyword in q_norm:
            # Flexible pattern allowing up to 25 intervening characters
            neg_regex = rf"(?:khong\s+muon|tranh|ngoai\s+tru|chan|khong\s+thich|nhung\s+khong|tired\s+of|no|not|without|avoid|except|bored\s+of|sick\s+of).{{0,25}}{re.escape(keyword)}"
            if re.search(neg_regex, q_norm):
                if genre not in excluded:
                    excluded.append(genre)
    return excluded


def _extract_anchor_movie(query: str) -> Optional[str]:
    """Extract the positive anchor movie from a critiquing query."""
    patterns = [
        r"(?:liked|loved|enjoyed|watched) ([A-Z][^,;]+?) but",
        r"(?:like|love) ([A-Z][^,;]+?) but",
        r"([A-Z][A-Za-z\s']+) (?:is great|is good|was great|was good) but",
        r"(?:thích|yêu thích|đã xem|xem qua|thich) ([A-Z][^,;]+?) (?:nhưng|nhung)",
        r"([A-Z][^,;]+?) (?:rất hay|khá hay|tốt) (?:nhưng|nhung)",
    ]
    for pattern in patterns:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            anchor = m.group(1).strip()
            anchor = re.sub(r"^(?:phim|movie)\s+", "", anchor, flags=re.IGNORECASE)
            return anchor.strip()
    return None


# ---------------------------------------------------------------------------
# Main Classifier: classify_intent
# ---------------------------------------------------------------------------

def classify_intent(query: str, session: Optional[SessionContext] = None) -> IntentResult:
    """Classify the user's intent using Three-tier Architecture with Security Guardrails."""
    q = query.strip()
    q_norm = strip_accents(q)

    # ═════════════════════════════════════════════════════════════════════════
    # TẦNG 1: Pre-Router & Security Guardrails (< 1ms, Free)
    # ═════════════════════════════════════════════════════════════════════════
    if _is_security_or_jailbreak(q):
        return IntentResult(
            intent="out_of_scope",
            confidence="high",
            extracted_entities={"security_flag": True},
            raw_query=q,
        )

    if _is_greeting_or_courtesy(q):
        return IntentResult(
            intent="greeting",
            confidence="high",
            extracted_entities={},
            raw_query=q,
        )

    if _is_clarification(q):
        return IntentResult(
            intent="clarification",
            confidence="high",
            extracted_entities={},
            raw_query=q,
        )

    if _is_out_of_scope(q):
        return IntentResult(
            intent="out_of_scope",
            confidence="high",
            extracted_entities={},
            raw_query=q,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # TẦNG 2: Enhanced Domain Classifier (< 5ms)
    # ═════════════════════════════════════════════════════════════════════════

    # ── Dạng 3: Peer Analysis / Comparison on Specific Movie ─────────────────
    extracted_title = _extract_movie_title_from_peer_query(q)
    if extracted_title:
        resolved = resolve_movie_title(extracted_title)
        entities = {
            "raw_title": extracted_title,
            "movie_title": resolved[0] if resolved else extracted_title,
            "movieId": resolved[1] if resolved else None,
            "fuzzy_matched": resolved is not None and resolved[0].lower() != extracted_title.lower(),
            "found_in_dataset": resolved is not None,
        }
        return IntentResult(
            intent="peer_analysis",
            confidence="high" if resolved else "medium",
            extracted_entities=entities,
            raw_query=q,
        )

    # ── Dạng 4: Explanation (Follow-up reasoning without specific title) ─────
    has_explain_kw = any(kw in q_norm for kw in _EXPLANATION_KEYWORDS)
    if has_explain_kw:
        has_recs = bool(session and session.has_recommendations)
        return IntentResult(
            intent="explanation",
            confidence="high",
            extracted_entities={
                "has_recs": has_recs,
                "last_recs": session.last_recommendations if has_recs else [],
            },
            raw_query=q,
        )

    # ── Dạng 6: Blind Spot ──────────────────────────────────────────────────
    if any(kw in q_norm for kw in _BLIND_SPOT_KEYWORDS):
        return IntentResult(
            intent="blind_spot",
            confidence="high",
            extracted_entities={},
            raw_query=q,
        )

    # ── Dạng 5: Critiquing ──────────────────────────────────────────────────
    if any(kw in q_norm for kw in _CRITIQUING_KEYWORDS):
        exclude_genres = _extract_exclude_genres(q)
        raw_anchor = _extract_anchor_movie(q)
        resolved_tup = resolve_movie_title(raw_anchor) if raw_anchor else None
        resolved_anchor = resolved_tup[0] if resolved_tup else None
        return IntentResult(
            intent="critiquing",
            confidence="high" if (exclude_genres or raw_anchor) else "medium",
            extracted_entities={
                "exclude_genres": exclude_genres,
                "anchor_movie": resolved_anchor or raw_anchor,
                "raw_anchor": raw_anchor,
            },
            raw_query=q,
        )

    # ── Dạng 2: Content Search ──────────────────────────────────────────────
    content_match_count = sum(1 for kw in _CONTENT_SEARCH_KEYWORDS if kw in q_norm)
    if content_match_count >= 1:
        personal_match_count = sum(1 for kw in _PERSONAL_REC_KEYWORDS if kw in q_norm)
        if content_match_count > personal_match_count:
            return IntentResult(
                intent="content_search",
                confidence="high" if content_match_count >= 2 else "medium",
                extracted_entities={"search_query": q},
                raw_query=q,
            )

    # ── Dạng 1: Personal Recommendation (catch-all default) ─────────────────
    return IntentResult(
        intent="personal_recommendation",
        confidence="medium",
        extracted_entities={},
        raw_query=q,
    )
