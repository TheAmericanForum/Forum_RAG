"""Agentic query agent: Opus 4.8 plans retrieval via a search tool, then writes a
cited answer using Anthropic's native Citations over the retrieved passages.

`answer(...)` is a generator yielding events consumed by both the CLI and the web API:
  {"type": "progress", "message": str}
  {"type": "token",    "text": str}
  {"type": "citation", "cited_text": str, "source": {...}, "index": int}
  {"type": "done",     "sources": [{"cited_text": str, "source": {...}, "index": int}, ...]}

"index" is a 1-based footnote number, stable per distinct (source, cited_text) pair so
repeated citations of the same passage reuse the same footnote.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Optional

from .config import get_settings
from .embed import embed_query
from .errors import ExternalServiceError
from . import store

log = logging.getLogger(__name__)

_client = None


def _client_():
    global _client
    if _client is None:
        import httpx
        from anthropic import Anthropic

        # Force IPv4: some Heroku dynos have broken outbound IPv6 routing, which
        # makes every connection fail deterministically even though DNS resolves
        # fine and retries don't help. See sc_rag/embed.py for the same fix.
        http_client = httpx.Client(transport=httpx.HTTPTransport(local_address="0.0.0.0"))
        _client = Anthropic(api_key=get_settings().require_anthropic_key(), http_client=http_client)
    return _client


SEARCH_TOOL = {
    "name": "search_transcripts",
    "description": (
        "Semantic search over community policy-discussion transcript passages, scoped to "
        "the conversation's policy area. Call it repeatedly with different queries to "
        "gather ALL relevant passages — e.g. one search per sub-topic, proposal, or "
        "concern. Optionally filter by session or speaker."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "session": {"type": "string", "description": "Restrict to a session label."},
            "speaker": {"type": "string", "description": "Restrict to a speaker label, e.g. S3."},
            "top_k": {"type": "integer", "description": "Max results (default 8)."},
        },
        "required": ["query"],
    },
}

RETRIEVE_SYSTEM = (
    "You are the retrieval planner for a RAG system over community policy-discussion "
    "transcripts. Your ONLY job right now is to gather evidence: call search_transcripts "
    "as many times as needed to collect every passage relevant to the user's question, "
    "within the conversation's policy area — vary your queries by sub-topic, proposal, "
    "and trade-off, but do not search outside that area. Do NOT write the final answer "
    "yet. When you have gathered enough, stop calling tools and reply with the single "
    "word DONE."
)

SYNTH_SYSTEM = (
    "You answer questions about community policy discussions using ONLY the provided "
    "transcript excerpts. Center the response on the PROPOSALS raised and the TRADE-OFFS "
    "and concerns discussed, and highlight where there is CONSENSUS and where there is "
    "DISAGREEMENT.\n\n"
    "WRITING STYLE\n"
    "- Write in clear, professional, grammatically correct English. The excerpts are "
    "machine-generated transcripts and may contain transcription errors, filler, or "
    "awkward, stilted, or archaic wording. When you state participants' points in your "
    "own words, render them in clean, natural prose — never reproduce transcription "
    "artifacts or unnatural phrasing (e.g. write \"new revenue\" or \"funding,\" not "
    "\"new monies\"). Silently correct obvious errors when you paraphrase.\n\n"
    "QUOTATIONS\n"
    "- Quote sparingly. Only quote when a participant's specific wording is itself "
    "notable; paraphrase most points.\n"
    "- Whenever you reproduce a participant's exact words, enclose them in double "
    "quotation marks. Never present someone's exact words without quotation marks.\n"
    "- Only quote a passage verbatim if it is already clean and grammatical. If a passage "
    "is garbled by transcription errors, paraphrase it in clean prose instead of quoting "
    "it — but still cite it, so the exact source appears in the footnote.\n\n"
    "CITATIONS\n"
    "- Support every factual claim drawn from the transcripts with a citation to the exact "
    "supporting passage, whether you quote it or paraphrase it.\n"
    "- If the excerpts do not cover something, say so rather than guessing.\n\n"
    "ATTRIBUTION\n"
    "- Never refer to a participant by their speaker label (e.g. \"S1\", \"S3\") in the body "
    "of your answer — those labels are internal IDs, not how a reader should see a person "
    "described. Use natural language instead: \"one resident said,\" \"another participant "
    "noted,\" \"a board member countered,\" \"several attendees agreed,\" etc.\n"
    "- The footnote citations themselves will still show which speaker said what — you do "
    "not need to (and should not) embed the speaker label in the sentence to compensate."
)


def _fmt_ms(ms: Any) -> str:
    s = int(ms or 0) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def _source_meta(ch: Optional[dict]) -> Optional[dict]:
    if not ch:
        return None
    return {
        "chunk_id": ch.get("chunk_id"),
        "session": ch.get("session"),
        "table": ch.get("table"),
        "date": ch.get("date"),
        "speakers": ch.get("speakers"),
        "time": _fmt_ms(ch.get("start_ms", 0)),
        "turn_start": ch.get("turn_start"),
        "turn_end": ch.get("turn_end"),
    }


def _doc_title(ch: dict) -> str:
    speakers = ", ".join(ch.get("speakers") or [])
    table = f"table {ch['table']}" if ch.get("table") else ""
    loc = " · ".join(p for p in [ch.get("session"), table, ch.get("date")] if p)
    turns = f"turns {ch.get('turn_start')}-{ch.get('turn_end')}"
    return f"{loc} · {speakers} · {_fmt_ms(ch.get('start_ms', 0))} ({turns})"


def _run_search(args: dict, gathered: dict[str, dict], *, policy_area: str) -> list[dict]:
    qv = embed_query(args["query"])
    results = store.search(
        qv,
        top_k=int(args.get("top_k") or get_settings().retrieval.top_k),
        policy_area=policy_area,
        session=args.get("session"),
        speaker=args.get("speaker"),
    )
    log.debug("Search %r returned %d result(s)", args.get("query"), len(results))
    brief = []
    for r in results:
        gathered[r["chunk_id"]] = r  # keep full payload (incl. text) for synthesis
        brief.append(
            {
                "chunk_id": r["chunk_id"],
                "session": r.get("session"),
                "table": r.get("table"),
                "speakers": r.get("speakers"),
                "time": _fmt_ms(r.get("start_ms", 0)),
                "policy_areas": r.get("policy_areas"),
                "snippet": (r.get("text", "") or "")[:300],
            }
        )
    return brief


def _synthesize(question: str, chunks: list[dict], *, policy_area: str) -> Iterator[dict]:
    """Stream a cited answer; yields token and citation events."""
    settings = get_settings()
    content: list[dict] = []
    docs: list[dict] = []  # aligned to citation.document_index
    for ch in chunks:
        content.append(
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": ch.get("text", "") or "",
                },
                "title": _doc_title(ch),
                "citations": {"enabled": True},
            }
        )
        docs.append(ch)

    q = f"{question}\n\n(Policy area: {policy_area})"
    content.append(
        {
            "type": "text",
            "text": (
                f"Question: {q}\n\nAnswer using the excerpts above. Cite the exact "
                "supporting passage for every claim. Quote only clean, notable wording "
                "(in double quotation marks) and paraphrase everything else in clear, "
                "professional prose."
            ),
        }
    )

    footnotes: dict[tuple, int] = {}  # (chunk_id, cited_text) -> 1-based footnote number
    answer_chars = 0  # running length of answer text streamed so far (footnote anchor)
    try:
        with _client_().messages.stream(
            model=settings.models.agent,
            max_tokens=8000,
            system=SYNTH_SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            messages=[{"role": "user", "content": content}],
        ) as stream:
            for event in stream:
                if event.type != "content_block_delta":
                    continue
                delta = event.delta
                dtype = getattr(delta, "type", None)
                if dtype == "text_delta":
                    answer_chars += len(delta.text)
                    yield {"type": "token", "text": delta.text}
                elif dtype == "citations_delta":
                    cit = getattr(delta, "citation", None)
                    idx = getattr(cit, "document_index", None)
                    src = docs[idx] if isinstance(idx, int) and 0 <= idx < len(docs) else None
                    cited_text = getattr(cit, "cited_text", "") or ""
                    key = (src.get("chunk_id") if src else idx, cited_text)
                    if key not in footnotes:
                        footnotes[key] = len(footnotes) + 1
                    # Anchor to answer_chars right now — the cited span has already
                    # streamed as text by the time this delta arrives, so this is the
                    # true position of the claim, not just the position at block end.
                    yield {
                        "type": "citation",
                        "cited_text": cited_text,
                        "source": _source_meta(src),
                        "index": footnotes[key],
                        "pos": answer_chars,
                    }
    except Exception as e:
        log.error("Synthesis stream failed: %s", e)
        raise ExternalServiceError(f"Anthropic answer-synthesis request failed: {e}") from e


def answer(
    question: str,
    *,
    policy_area: str,
    session: Optional[str] = None,
    speaker: Optional[str] = None,
    max_rounds: int = 6,
) -> Iterator[dict]:
    settings = get_settings()
    client = _client_()
    gathered: dict[str, dict] = {}

    base = f"{question}\n\n(Policy area: {policy_area})"
    messages: list[dict] = [{"role": "user", "content": base}]

    yield {"type": "progress", "message": "Searching transcripts…"}

    for round_ in range(max_rounds):
        try:
            resp = client.messages.create(
                model=settings.models.agent,
                max_tokens=4000,
                system=RETRIEVE_SYSTEM,
                tools=[SEARCH_TOOL],
                tool_choice={"type": "any"} if round_ == 0 else {"type": "auto"},
                output_config={"effort": "low"},
                messages=messages,
            )
        except Exception as e:
            log.error("Retrieval planner call failed (round %d): %s", round_, e)
            raise ExternalServiceError(f"Anthropic retrieval-planning request failed: {e}") from e

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": resp.content})
        if not tool_uses:
            break

        tool_results = []
        for tu in tool_uses:
            args = dict(tu.input)
            # Apply UI-provided filters as defaults the model didn't override.
            if session and not args.get("session"):
                args["session"] = session
            if speaker and not args.get("speaker"):
                args["speaker"] = speaker
            yield {"type": "progress", "message": f"🔎 {args.get('query', '')}"}
            try:
                brief = _run_search(args, gathered, policy_area=policy_area)
            except Exception as e:
                # Don't let one bad search (e.g. a transient embedding/Qdrant error)
                # abort the whole answer — tell the model the search failed and move on.
                log.warning("Search failed for query %r: %s", args.get("query"), e)
                yield {"type": "progress", "message": f"search failed: {e}"}
                brief = []
            tool_results.append(
                {"type": "tool_result", "tool_use_id": tu.id, "content": json.dumps(brief)}
            )
        messages.append({"role": "user", "content": tool_results})

    chunks = list(gathered.values())
    if not chunks:
        yield {"type": "token", "text": "I couldn't find any relevant passages in the indexed transcripts."}
        yield {"type": "done", "sources": []}
        return

    yield {"type": "progress", "message": f"Found {len(chunks)} passages. Composing cited answer…"}

    collected: dict[int, dict] = {}
    for ev in _synthesize(question, chunks, policy_area=policy_area):
        if ev["type"] == "citation" and ev["index"] not in collected:
            collected[ev["index"]] = {
                "cited_text": ev["cited_text"],
                "source": ev["source"],
                "index": ev["index"],
            }  # "pos" is per-occurrence and intentionally not carried into `done.sources`
        yield ev

    sources = [collected[i] for i in sorted(collected)]
    yield {"type": "done", "sources": sources}
