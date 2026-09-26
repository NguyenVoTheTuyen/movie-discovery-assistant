"""
cli.py — Rich CLI Chat Interface for TrustedAI Movie Assistant.

Usage:
    python src/cli.py
    python src/cli.py --offline       # Use cached LLM responses
    python src/cli.py --user 1        # Start with specific user

Commands:
    /reset       — Clear conversation history for current user
    /switch <id> — Switch to a different user
    /whoami      — Show current user profile
    /quit | /exit | exit | quit — Exit the app
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure src/ is on path when running from Exam/
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich import box

from data_loader import (
    get_user_rating_count,
    get_user_top_genres,
    get_user_top_rated_movies,
    verify_data_loaded,
)
from session_manager import session_manager

console = Console()

# ---------------------------------------------------------------------------
# Branding
# ---------------------------------------------------------------------------

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🎬  TrustedAI — Movie Discovery Assistant           ║
║          Powered by Gemini 2.5 Flash + Two-Stage Hybrid      ║
║          5,135 movies | 74,064 ratings | 610 users           ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
**Available Commands:**
| Command | Description |
|---------|-------------|
| `/reset` | Clear your conversation history |
| `/switch <userId>` | Switch to a different user (e.g. `/switch 15`) |
| `/whoami` | Show your current profile |
| `/help` | Show this help |
| `/quit` | Exit the app |

**Query Types Supported:**
1. 🎯 *Personal*: "What should I watch tonight?"
2. 🔍 *Search*: "I want a dark psychological thriller with a twist"
3. 👥 *Peer*: "What do people like me think about Pulp Fiction?"
4. 💬 *Explain*: "Why do you think I'd like that?"
5. 🚫 *Critiquing*: "I liked Toy Story but I'm tired of animated movies"
6. 🗺️ *Blind Spot*: "What genres am I missing?"
"""


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_banner(offline: bool = False) -> None:
    style = "bold magenta" if not offline else "bold yellow"
    console.print(BANNER, style=style)
    if offline:
        console.print(
            Panel("⚠️  OFFLINE MODE — Using cached LLM responses", style="yellow", box=box.ROUNDED)
        )


def print_user_profile(user_id: int) -> None:
    count = get_user_rating_count().get(user_id, 0)
    genres = get_user_top_genres(user_id, top_n=5)
    top_rated = get_user_top_rated_movies(user_id, min_rating=4.5, top_n=3)

    if count == 0:
        path = "🆕 Cold-Start (Bayesian)"
    elif count < 20:
        path = "📄 Content-Based (TF-IDF)"
    else:
        path = "🔀 Hybrid (CF + Content + Recency)"

    table = Table(box=box.ROUNDED, border_style="cyan", show_header=False)
    table.add_column("Key", style="bold cyan", width=20)
    table.add_column("Value", style="white")
    table.add_row("User ID", str(user_id))
    table.add_row("Total Ratings", str(count))
    table.add_row("Active Path", path)
    table.add_row("Top Genres", ", ".join(genres) if genres else "—")
    table.add_row(
        "Top Rated (4.5★+)",
        "\n".join(f"{m['title']} ({m['rating']}★)" for m in top_rated) if top_rated else "—"
    )
    console.print(Panel(table, title=f"👤 User Profile", border_style="cyan"))


def print_intent_badge(intent: str, confidence: str, path: str) -> None:
    intent_icons = {
        "personal_recommendation": "🎯",
        "content_search": "🔍",
        "peer_analysis": "👥",
        "explanation": "💬",
        "critiquing": "🚫",
        "blind_spot": "🗺️",
    }
    icon = intent_icons.get(intent, "❓")
    conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(confidence, "white")

    parts = Text()
    parts.append(f"{icon} {intent}", style="bold white")
    parts.append(f"  ·  confidence: ", style="dim")
    parts.append(confidence, style=f"bold {conf_color}")
    parts.append(f"  ·  path: ", style="dim")
    parts.append(path, style="bold blue")
    console.print(parts)


def print_response(response_text: str) -> None:
    """Render the LLM response as markdown in a styled panel."""
    md = Markdown(response_text)
    console.print(Panel(md, title="🤖 TrustedAI", border_style="green", box=box.ROUNDED))


def print_thinking() -> None:
    console.print("  [dim]🧠 Thinking...[/dim]")


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

def run_chat(user_id: int, offline: bool = False) -> None:
    """Main interactive chat loop."""
    if offline:
        os.environ["OFFLINE_MODE"] = "true"

    # Late import after env setup
    from assistant_agent import process_query

    print_banner(offline=offline)

    console.print(f"\n[dim]Loading dataset and pre-computing matrices...[/dim]")
    try:
        verify_data_loaded()
    except Exception as e:
        console.print(f"[red]❌ Failed to load data: {e}[/red]")
        console.print("[yellow]Check that data/ directory exists with ml-latest-small-filtered/ contents.[/yellow]")
        sys.exit(1)

    console.print()
    print_user_profile(user_id)
    console.print(f"\n[dim]Type [bold]/help[/bold] for commands or start chatting![/dim]\n")

    while True:
        try:
            console.print(Rule(style="dim"))
            raw_input = Prompt.ask(f"[bold magenta]You[/bold magenta] [dim](User {user_id})[/dim]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! 🎬[/dim]")
            break

        query = raw_input.strip()

        # ── Built-in commands ──────────────────────────────────────────────
        if not query:
            continue

        if query.lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("[dim]Goodbye! 🎬[/dim]")
            break

        if query.lower() == "/reset":
            session_manager.reset(user_id)
            console.print("[yellow]✓ Conversation reset.[/yellow]")
            continue

        if query.lower() == "/help":
            console.print(Markdown(HELP_TEXT))
            continue

        if query.lower() == "/whoami":
            print_user_profile(user_id)
            continue

        if query.lower().startswith("/switch"):
            parts = query.split()
            if len(parts) == 2 and parts[1].isdigit():
                new_uid = int(parts[1])
                rating_counts = get_user_rating_count()
                if new_uid not in rating_counts and new_uid != 999:
                    console.print(f"[yellow]⚠ User {new_uid} not in dataset. Valid range: 1–610 (or 999 for cold-start demo).[/yellow]")
                else:
                    user_id = new_uid
                    console.print(f"[green]✓ Switched to User {user_id}.[/green]")
                    print_user_profile(user_id)
            else:
                console.print("[red]Usage: /switch <userId>[/red]")
            continue

        # ── Process query ──────────────────────────────────────────────────
        print_thinking()
        t0 = time.perf_counter()

        try:
            result = process_query(user_id, query, offline=offline)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

        elapsed = time.perf_counter() - t0

        console.print()
        print_intent_badge(
            intent=result["intent"],
            confidence=result["confidence"],
            path=result["path_used"],
        )
        console.print(f"[dim]  ⏱ {elapsed:.2f}s[/dim]")
        console.print()
        print_response(result["response"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TrustedAI Movie Discovery Assistant")
    parser.add_argument(
        "--user", type=int, default=1,
        help="Starting user ID (default: 1). Use 999 for cold-start demo."
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Run in offline mode using cached LLM responses (no API key needed)."
    )
    args = parser.parse_args()
    run_chat(user_id=args.user, offline=args.offline)


if __name__ == "__main__":
    main()
