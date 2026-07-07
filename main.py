"""CLI query client (local dev). The deployed interface is the FastAPI app in app.py.

Usage:
  python main.py "What trade-offs were raised about <area>?"
  python main.py --policy-area "Housing" "What proposals came up?"
  python main.py                      # interactive REPL
"""
from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console

from forum_rag.agent import answer
from forum_rag.errors import ConfigError, ExternalServiceError
from forum_rag.logging import setup_logging

log = logging.getLogger(__name__)


def _ask(console: Console, question: str, args) -> None:
    """Stream one answer to `question`, printing tokens live and sources at the end."""
    sources: list[dict] = []
    try:
        for ev in answer(
            question,
            policy_area=args.policy_area,
            session=args.session,
            speaker=args.speaker,
        ):
            kind = ev["type"]
            if kind == "progress":
                if args.show_progress:
                    console.print(f"[dim]{ev['message']}[/]")
            elif kind == "token":
                sys.stdout.write(ev["text"])
                sys.stdout.flush()
            elif kind == "done":
                sources = ev["sources"]
    except (ConfigError, ExternalServiceError) as e:
        console.print(f"\n[bold red]Error:[/] {e}")
        return
    except Exception:
        log.exception("Unexpected error answering question: %r", question)
        console.print("\n[bold red]Unexpected error.[/] See logs/forum_rag.log for details.")
        return

    sys.stdout.write("\n")
    if sources:
        console.print("\n[bold]Sources[/]")
        for i, citation in enumerate(sources, 1):
            source_meta = citation.get("source") or {}
            speakers = ", ".join(source_meta.get("speakers") or [])
            table = f"table {source_meta['table']}" if source_meta.get("table") else ""
            location = " · ".join(
                p for p in [source_meta.get("session"), table, source_meta.get("date")] if p
            )
            console.print(
                f"[cyan][{i}][/] {location} · {speakers} · {source_meta.get('time')} "
                f"(turns {source_meta.get('turn_start')}-{source_meta.get('turn_end')})"
            )
            quote = (citation.get("cited_text") or "").strip()
            if quote:
                console.print(f'    [italic dim]"{quote}"[/]')


def _repl(console: Console, args) -> None:
    """Interactive loop: read a question, answer it, repeat until Ctrl-C/EOF."""
    console.print("[bold]Policy-discussion RAG[/] — ask a question (Ctrl-C to exit).")
    try:
        while True:
            user_input = console.input("\n[bold green]?[/] ").strip()
            if user_input:
                _ask(console, user_input, args)
    except (KeyboardInterrupt, EOFError):
        console.print("\nBye.")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Query the policy-discussion RAG.")
    parser.add_argument("question", nargs="*", help="Question to ask (omit for interactive mode).")
    parser.add_argument("--policy-area", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--show-progress", action="store_true", help="Print retrieval progress.")
    args = parser.parse_args()

    console = Console()
    question = " ".join(args.question).strip()
    if question:
        _ask(console, question, args)
    else:
        _repl(console, args)


if __name__ == "__main__":
    main()
