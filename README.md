# TrustedAI — Movie Discovery Assistant

> 📌 **Đề bài & Yêu cầu gốc của bài test:** Toàn bộ nội dung yêu cầu kiểm tra ban đầu được lưu trữ nguyên vẹn tại [README_ORIGINAL.md](./README_ORIGINAL.md).
>
> **AI-powered movie recommendation system** built on MovieLens (5,135 movies, 74,064 ratings, 610 users).
> Combines Collaborative Filtering, TF-IDF Content Similarity, Bayesian Rating, and Gemini 2.5 Flash LLM for grounded, explainable recommendations.

---

## Architecture Overview

```
Two-Stage Adaptive Hybrid (Plan A):

    Stage 1 — Candidate Retrieval (based on user history count):
        ├── 0 ratings  → Bayesian Weighted Rating (cold-start)
        ├── < 20 ratings → TF-IDF Content Similarity
        └── ≥ 20 ratings → Hybrid (CF + Content + Recency)

    Stage 2 — LLM Re-ranking:
        └── Gemini 2.5 Flash → picks Top 1–3 + grounded explanation
```

**6 Supported Query Types:**

| # | Intent | Example |
|---|--------|---------|
| 1 | Personal Recommendation | "What should I watch tonight?" |
| 2 | Content Search | "I want a dark psychological thriller with a twist" |
| 3 | Peer Analysis | "What do people like me think about Pulp Fiction?" |
| 4 | Explanation (multi-turn) | "Why do you think I'd like that?" |
| 5 | Critiquing | "I liked Toy Story but I'm tired of animated movies" |
| 6 | Blind Spot | "What genres am I missing?" |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Gemini API Key (free tier — get at https://aistudio.google.com/app/apikey)

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env and set: GEMINI_API_KEY=your_key_here

# 3. Run the assistant
python src/cli.py
```

### Offline Mode (No API Key Required)

```bash
# Read pre-logged sample conversations
cat output/sample_conversations.md

# Or run in offline mode (engines work, LLM responses are cached)
python src/cli.py --offline
```

### Web Application (Interactive Modern UI)

```bash
# Run web server
python src/web_app.py

# Open browser at:
# http://127.0.0.1:8000
```
- Features: Glassmorphic dark UI, real-time typing indicators, user taste inspector, offline mode toggle, quick query chips, and bilingual Vietnamese/English support.

### Running Automated Test Suites

```bash
# 1. 200 QA Test Suite across 10 categories (Happy Path, Typos, Security, Cold-start, etc.)
python scripts/test_200_qa_suite.py

# 2. Objective Sample Prompts Suite across user tiers + full 610-user algorithmic check
python scripts/test_sample_prompts_suite.py
```

### Start with a Specific User in CLI

```bash
python src/cli.py --user 1    # Rich history (190 ratings) → Hybrid path
python src/cli.py --user 30   # Sparse history (18 ratings) → Content path
python src/cli.py --user 999  # Cold start (0 ratings) → Bayesian path
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `/reset` | Clear conversation history |
| `/switch <id>` | Switch to different user |
| `/whoami` | Show user profile |
| `/help` | Show help |
| `/quit` | Exit |

---

## Project Structure

```
Exam/
├── data/ml-latest-small-filtered/
│   ├── movies_with_plots.csv   # 5,135 movies with plots
│   ├── ratings.csv             # 74,064 ratings
│   ├── tags.csv                # 2,440 tags
│   └── movies.csv, links.csv
├── src/
│   ├── data_loader.py          # Load & cache DataFrames
│   ├── bayesian_engine.py      # Bayesian Weighted Rating (m=10)
│   ├── content_engine.py       # TF-IDF Content Similarity
│   ├── cf_engine.py            # CF (Cosine K=20) + fillna(0) fix
│   ├── hybrid_engine.py        # Combine + T_half floor + Diversity
│   ├── intent_classifier.py    # Two-tier intent detection
│   ├── session_manager.py      # Multi-turn conversation memory
│   ├── assistant_agent.py      # LLM Agent + 6 handlers
│   ├── web_app.py              # FastAPI web server (http://127.0.0.1:8000)
│   └── cli.py                  # Rich CLI interface
├── evaluation/
│   ├── offline_eval.py         # LOO + Hit@K evaluation
│   └── baseline_eval.py        # Bayesian m sensitivity + baselines
├── scripts/
│   ├── test_200_qa_suite.py    # 200 automated QA test cases
│   └── test_sample_prompts_suite.py  # Sample prompts audit
├── output/
│   ├── sample_conversations.md # 6 pre-logged test cases
│   ├── qa_200_test_results.json    # Full 200-test results
│   └── sample_prompts_audit_results.json
├── .env.example
├── requirements.txt
├── REPORT.md                   # English technical report
└── REPORT_VN.md                # Vietnamese technical report
```

---

## Running Evaluation

```bash
# Leave-One-Out evaluation + Hit@K + diversity metrics
python evaluation/offline_eval.py

# Bayesian m sensitivity analysis + baseline comparison
python evaluation/baseline_eval.py
```

---

## Key Design Decisions

| Decision | Alternative | Rationale |
|----------|-------------|-----------|
| Two-Stage Hybrid | Pure RAG | RAG lacks peer grounding → risk of recommending low-rated films |
| TF-IDF Cosine | SentenceTransformers | Zero infra, no 500MB model download, sufficient for 5K movies |
| Gemini 2.5 Flash | GPT-4o-mini | Free tier, 1M context window, grounded explanations |
| m=10 Bayesian | m=30 | m=30 → only 13.7% catalog eligible; m=10 → 32.4% (better coverage) |
| T_half floor 30 days | Raw median delta | User 1 & 30 rated all movies in 1 day → degenerate recency without floor |

---

## Dataset Facts

| Metric | Value |
|--------|-------|
| Movies | 5,135 (100% have plot) |
| Ratings | 74,064 (610 users) |
| Matrix sparsity | 2.36% filled |
| Tags coverage | 20% of movies (1,026) |
| Rating scale | 0.5–5.0 (half-star) |
| Global mean | 3.4947 |
| Date range | 1996-03-29 → 2018-09-24 |
