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

log = logging.getLogger(__name__)


def _ask(console: Console, question: str, args) -> None:
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
        console.print("\n[bold red]Unexpected error.[/] See logs/sc_rag.log for details.")
        return

    sys.stdout.write("\n")
    if sources:
        console.print("\n[bold]Sources[/]")
        for i, s in enumerate(sources, 1):
            src = s.get("source") or {}
            speakers = ", ".join(src.get("speakers") or [])
            table = f"table {src['table']}" if src.get("table") else ""
            loc = " · ".join(p for p in [src.get("session"), table, src.get("date")] if p)
            console.print(
                f"[cyan][{i}][/] {loc} · {speakers} · {src.get('time')} "
                f"(turns {src.get('turn_start')}-{src.get('turn_end')})"
            )
            quote = (s.get("cited_text") or "").strip()
            if quote:
                console.print(f'    [italic dim]"{quote}"[/]')


def _repl(console: Console, args) -> None:
    console.print("[bold]Policy-discussion RAG[/] — ask a question (Ctrl-C to exit).")
    try:
        while True:
            q = console.input("\n[bold green]?[/] ").strip()
            if q:
                _ask(console, q, args)
    except (KeyboardInterrupt, EOFError):
        console.print("\nBye.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Query the policy-discussion RAG.")
    ap.add_argument("question", nargs="*", help="Question to ask (omit for interactive mode).")
    ap.add_argument("--policy-area", default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--speaker", default=None)
    ap.add_argument("--show-progress", action="store_true", help="Print retrieval progress.")
    args = ap.parse_args()

    console = Console()
    question = " ".join(args.question).strip()
    if question:
        _ask(console, question, args)
    else:
        _repl(console, args)


if __name__ == "__main__":
    main()
