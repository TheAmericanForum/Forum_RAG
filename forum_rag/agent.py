"""Agentic query agent: Sonnet plans retrieval via a search tool, then Opus writes a
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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterator, Optional

from anthropic.types import TextBlockParam
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import get_settings
from .embed import embed_query
from .errors import ExternalServiceError, is_retryable_api_error
from . import store

log = logging.getLogger(__name__)

_client = None


def _get_client():
    """Return the process-wide Anthropic client, creating it on first use."""
    global _client
    if _client is None:
        import httpx
        from anthropic import Anthropic

        # Force IPv4: some Heroku dynos have broken outbound IPv6 routing, which
        # makes every connection fail deterministically even though DNS resolves
        # fine and retries don't help. See forum_rag/embed.py for the same fix.
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
    # Identical on every call, so mark it as the end of the cacheable system+tools
    # prefix shared across every retrieval-planner request.
    "cache_control": {"type": "ephemeral"},
}

RETRIEVE_SYSTEM = (
    "You are the retrieval planner for a RAG system over community policy-discussion "
    "transcripts. Your ONLY job right now is to gather evidence: call search_transcripts "
    "as many times as needed to collect every passage relevant to the user's LATEST "
    "question, within the conversation's policy area — vary your queries by sub-topic, "
    "proposal, and trade-off, but do not search outside that area. Do NOT write the "
    "final answer yet. When you have gathered enough, stop calling tools and reply with "
    "the single word DONE.\n\n"
    "CONVERSATION CONTEXT\n"
    "- You may be shown earlier turns of this conversation before the latest question. "
    "Use them only to resolve references in the latest question (e.g. \"that proposal,\" "
    "\"the second one,\" \"what about cost?\") into concrete search terms.\n"
    "- Search for what the latest question needs. Don't re-run searches for ground "
    "already covered in an earlier turn unless the latest question asks about it again."
)

SYNTH_SYSTEM = (
    "You answer questions about community policy discussions using ONLY the provided "
    "transcript excerpts. Center the response on the PROPOSALS raised and the TRADE-OFFS "
    "and concerns discussed, and highlight where there is CONSENSUS and where there is "
    "DISAGREEMENT.\n\n"
    "CONVERSATION CONTEXT\n"
    "- You may be shown earlier turns of this conversation before the current question. "
    "Answer as a natural continuation of that dialogue: resolve pronouns and references "
    "the way a person following along would, and don't re-explain something already "
    "covered unless the current question asks for it again.\n"
    "- Citations are still required for every claim drawn from the excerpts provided in "
    "THIS turn. Prior answers in the conversation are context, not a source you cite.\n\n"
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


def _cached_system(text: str) -> list[TextBlockParam]:
    """Wrap a system prompt as a single cacheable block."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


# Prior exchanges kept for conversational context, most recent first when trimming.
# Bounds prompt size/cost on long-running conversations; older turns are dropped since
# the model only needs enough context to resolve references in the current question.
MAX_HISTORY_EXCHANGES = 6


def _history_messages(history: Optional[list[dict]]) -> list[dict]:
    """Turn prior conversation turns into plain user/assistant messages, keeping only
    the most recent MAX_HISTORY_EXCHANGES exchanges."""
    if not history:
        return []
    trimmed = history[-MAX_HISTORY_EXCHANGES * 2 :]
    return [{"role": turn["role"], "content": turn["content"]} for turn in trimmed]


def _fmt_ms(ms: Any) -> str:
    total_seconds = int(ms or 0) // 1000
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def _source_meta(chunk: Optional[dict]) -> Optional[dict]:
    """Build the citation-facing metadata dict for one retrieved chunk (or None)."""
    if not chunk:
        return None
    return {
        "chunk_id": chunk.get("chunk_id"),
        "citation_id": chunk.get("citation_id"),
        "session": chunk.get("session"),
        "table": chunk.get("table"),
        "date": chunk.get("date"),
        "speakers": chunk.get("speakers"),
        "time": _fmt_ms(chunk.get("start_ms", 0)),
        "turn_start": chunk.get("turn_start"),
        "turn_end": chunk.get("turn_end"),
    }


def _doc_title(chunk: dict) -> str:
    """Human-readable label shown to the model for a passage, e.g. in citation grounding."""
    speakers = ", ".join(chunk.get("speakers") or [])
    table = f"table {chunk['table']}" if chunk.get("table") else ""
    location = " · ".join(p for p in [chunk.get("session"), table, chunk.get("date")] if p)
    turns = f"turns {chunk.get('turn_start')}-{chunk.get('turn_end')}"
    return f"{location} · {speakers} · {_fmt_ms(chunk.get('start_ms', 0))} ({turns})"


def _do_search(
    dispatch_idx: int,
    tool_use_id: str,
    args: dict,
    gathered: dict[str, dict],
    *,
    policy_area: str,
) -> tuple[int, str, list, Any]:
    """Run one search_transcripts tool call, converting any failure into an error
    value instead of raising, so one failed search doesn't abort the whole round.

    `dispatch_idx` is this call's position in the batch of tool calls submitted to
    the ThreadPoolExecutor this round; since they complete out of order (via
    as_completed), the caller uses it to write each result back into a
    same-length, order-preserving list.
    """
    try:
        brief = _run_search(args, gathered, policy_area=policy_area)
    except Exception as e:
        log.warning("Search failed for query %r: %s", args.get("query"), e)
        return dispatch_idx, tool_use_id, [], e
    return dispatch_idx, tool_use_id, brief, None


def _run_search(args: dict, gathered: dict[str, dict], *, policy_area: str) -> list[dict]:
    """Embed one query, search the store, and merge results into `gathered`
    (a dict keyed by chunk_id, so the same chunk found by multiple queries across
    multiple rounds is only kept once). Returns a brief (truncated) result summary
    for the model — full chunk text stays in `gathered` for the synthesis step."""
    query_vector = embed_query(args["query"])
    results = store.search(
        query_vector,
        top_k=int(args.get("top_k") or get_settings().retrieval.top_k),
        policy_area=policy_area,
        session=args.get("session"),
        speaker=args.get("speaker"),
    )
    log.debug("Search %r returned %d result(s)", args.get("query"), len(results))
    brief = []
    for result in results:
        gathered[result["chunk_id"]] = result  # keep full payload (incl. text) for synthesis
        brief.append(
            {
                "chunk_id": result["chunk_id"],
                "session": result.get("session"),
                "table": result.get("table"),
                "speakers": result.get("speakers"),
                "time": _fmt_ms(result.get("start_ms", 0)),
                "policy_areas": result.get("policy_areas"),
                "snippet": (result.get("text", "") or "")[:300],
            }
        )
    return brief


def _synthesize(
    question: str,
    chunks: list[dict],
    *,
    policy_area: str,
    history_messages: Optional[list[dict]] = None,
) -> Iterator[dict]:
    """Stream a cited answer over `chunks`; yields token and citation events.

    `history_messages` (already-trimmed prior user/assistant turns, if any) is
    prepended so the answer reads as a continuation of the conversation rather than
    a one-off response.

    Retried only *before the first event is emitted*: the common failure here is a
    retryable error (e.g. 429/529 overloaded) while establishing the stream, and
    nothing has been forwarded to the caller yet, so re-establishing is safe. The
    final retry falls back to a smaller model (synthesis_agent_fallback) so a
    sustained overload of the primary degrades quality instead of failing. Once any
    token/citation has been yielded (to the browser via SSE), a mid-stream failure
    can't be retried without replaying already-shown output, so it ends the stream
    with an ExternalServiceError instead.
    """
    settings = get_settings()
    content: list[dict] = []
    docs: list[dict] = []  # aligned to citation.document_index
    for chunk in chunks:
        content.append(
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": chunk.get("text", "") or "",
                },
                "title": _doc_title(chunk),
                "citations": {"enabled": True},
            }
        )
        docs.append(chunk)

    question_with_area = f"{question}\n\n(Policy area: {policy_area})"
    content.append(
        {
            "type": "text",
            "text": (
                f"Question: {question_with_area}\n\nAnswer using the excerpts above. Cite the exact "
                "supporting passage for every claim. Quote only clean, notable wording "
                "(in double quotation marks) and paraphrase everything else in clear, "
                "professional prose."
            ),
        }
    )
    messages = [*(history_messages or []), {"role": "user", "content": content}]

    # Attempt plan: retry the primary synthesis model, then fall back to a smaller,
    # less capacity-constrained model if it stays overloaded. Fallback is skipped
    # when it's unset or identical to the primary (see synthesis_agent_fallback).
    primary = settings.models.synthesis_agent
    fallback = settings.models.synthesis_agent_fallback
    attempt_models = [primary, primary]
    if fallback and fallback != primary:
        attempt_models.append(fallback)
    max_attempts = len(attempt_models)
    for attempt, model in enumerate(attempt_models, 1):
        footnotes: dict[tuple, int] = {}  # (chunk_id, cited_text) -> 1-based footnote number
        answer_chars = 0  # running length of answer text streamed so far (footnote anchor)
        emitted = False  # whether any event has been yielded on this attempt
        try:
            with _get_client().messages.stream(
                model=model,
                max_tokens=8000,
                system=_cached_system(SYNTH_SYSTEM),
                messages=messages,
            ) as stream:
                for event in stream:
                    if event.type != "content_block_delta":
                        continue
                    delta = event.delta
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        answer_chars += len(delta.text)
                        emitted = True
                        yield {"type": "token", "text": delta.text}
                    elif delta_type == "citations_delta":
                        citation = getattr(delta, "citation", None)
                        doc_index = getattr(citation, "document_index", None)
                        source_chunk = (
                            docs[doc_index] if isinstance(doc_index, int) and 0 <= doc_index < len(docs) else None
                        )
                        if source_chunk is None:
                            # Anthropic returned a document_index we can't map back to a
                            # chunk — shouldn't happen given `docs` is built from the same
                            # `content` sent in this request, but if it does, the footnote
                            # below falls back to the raw (possibly None) doc_index as its
                            # dedup key instead of a stable chunk_id.
                            log.warning(
                                "citations_delta document_index %r out of range for %d document(s)",
                                doc_index, len(docs),
                            )
                        cited_text = getattr(citation, "cited_text", "") or ""
                        key = (source_chunk.get("chunk_id") if source_chunk else doc_index, cited_text)
                        if key not in footnotes:
                            footnotes[key] = len(footnotes) + 1
                        # Anchor to answer_chars right now — the cited span has already
                        # streamed as text by the time this delta arrives, so this is the
                        # true position of the claim, not just the position at block end.
                        emitted = True
                        yield {
                            "type": "citation",
                            "cited_text": cited_text,
                            "source": _source_meta(source_chunk),
                            "index": footnotes[key],
                            "pos": answer_chars,
                        }
            return  # stream completed successfully
        except Exception as e:
            # Only re-establish the stream if nothing has been forwarded to the caller
            # yet (else we'd replay shown output) and the error is worth retrying.
            if not emitted and attempt < max_attempts and is_retryable_api_error(e):
                backoff = min(8, 2 ** (attempt - 1))
                next_model = attempt_models[attempt]  # attempt is 1-based, so this is the next one
                log.warning(
                    "Synthesis stream on %s failed before any output (attempt %d/%d), "
                    "retrying in %ds on %s: %s",
                    model, attempt, max_attempts, backoff, next_model, e,
                )
                time.sleep(backoff)
                continue
            log.error("Synthesis stream failed: %s", e)
            raise ExternalServiceError(f"Anthropic answer-synthesis request failed: {e}") from e


@retry(
    # Fewer attempts and a shorter cap than embed.py/classify.py's background-job
    # retries: this call is in the critical path of a live, streamed user query, so
    # it's better to fail fast with a clear error than make someone wait a long time
    # for a retry chain to exhaust.
    retry=retry_if_exception(is_retryable_api_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=8),
    reraise=True,
)
def _call_retrieval_planner(client, settings, messages: list[dict], round_num: int):
    """Call the retrieval-planner model for one round of tool-calling."""
    try:
        return client.messages.create(
            model=settings.models.retrieval_agent,
            max_tokens=4000,
            system=_cached_system(RETRIEVE_SYSTEM),
            tools=[SEARCH_TOOL],
            tool_choice={"type": "any"} if round_num == 0 else {"type": "auto"},
            messages=messages,
        )
    except Exception as e:
        if is_retryable_api_error(e):
            log.warning("Retrieval planner call failed (round %d), will retry: %s", round_num, e)
            raise
        log.error("Retrieval planner call failed (round %d, non-retryable): %s", round_num, e)
        raise ExternalServiceError(f"Anthropic retrieval-planning request failed: {e}") from e


def answer(
    question: str,
    *,
    policy_area: str,
    session: Optional[str] = None,
    speaker: Optional[str] = None,
    max_rounds: int = 6,
    history: Optional[list[dict]] = None,
) -> Iterator[dict]:
    """Answer a question end to end: plan and run searches (tool-use loop, up to
    `max_rounds`), then stream a cited answer over whatever was found.

    `history` is prior turns of this conversation ({"role": "user"|"assistant",
    "content": str}, oldest first, not including `question`). It's trimmed to the
    most recent MAX_HISTORY_EXCHANGES and given to both the retrieval planner (to
    resolve references like "that" or "the second one" into search terms) and the
    synthesis step (to keep the answer conversational), so this is a chat, not a
    string of one-off lookups.

    Each round asks the retrieval-planner model to call search_transcripts zero or
    more times; `gathered` accumulates results in a dict keyed by chunk_id so the
    same chunk surfaced by different queries (within or across rounds) is only kept
    once. `any_search_errored` distinguishes "no results because nothing matched"
    from "no results because the search backend failed", which drives different
    user-facing messages if nothing was found by the time the loop ends.
    """
    settings = get_settings()
    client = _get_client()
    gathered: dict[str, dict] = {}

    history_messages = _history_messages(history)
    base = f"{question}\n\n(Policy area: {policy_area})"
    messages: list[dict] = [*history_messages, {"role": "user", "content": base}]
    any_search_errored = False
    # Tracks the most recently cache-marked message content block so each round's
    # cache breakpoint can be moved forward rather than left to accumulate — the
    # API allows at most 4 total (1 is already used by the static system+tools mark).
    cached_block: Optional[dict] = None

    yield {"type": "progress", "message": "Searching transcripts…"}

    for round_num in range(max_rounds):
        resp = _call_retrieval_planner(client, settings, messages, round_num)

        tool_uses = [block for block in resp.content if block.type == "tool_use"]
        messages.append({"role": "assistant", "content": resp.content})
        if not tool_uses:
            break

        # Normalize all tool call args up front so progress messages can be emitted
        # before dispatching searches, and apply UI-level session/speaker defaults.
        pending = []
        for tool_use_block in tool_uses:
            args = dict(tool_use_block.input)
            if session and not args.get("session"):
                args["session"] = session
            if speaker and not args.get("speaker"):
                args["speaker"] = speaker
            yield {"type": "progress", "message": f"🔎 {args.get('query', '')}"}
            pending.append((tool_use_block.id, args))

        # Run all searches for this round in parallel (embed + Qdrant per call).
        tool_results: list[dict] = [{} for _ in pending]

        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            futures = {
                pool.submit(_do_search, i, tool_use_id, args, gathered, policy_area=policy_area): i
                for i, (tool_use_id, args) in enumerate(pending)
            }
            for future in as_completed(futures):
                dispatch_idx, tool_use_id, brief, err = future.result()
                if err is not None:
                    yield {"type": "progress", "message": f"search failed: {err}"}
                    any_search_errored = True
                    brief = []
                tool_results[dispatch_idx] = {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(brief),
                }
        # Move the dynamic cache breakpoint to the end of this round's results, so
        # the next round's call reuses everything up through here from cache.
        if cached_block is not None:
            cached_block.pop("cache_control", None)
        tool_results[-1]["cache_control"] = {"type": "ephemeral"}
        cached_block = tool_results[-1]
        messages.append({"role": "user", "content": tool_results})

    rounds_used = round_num + 1
    chunks = list(gathered.values())
    if not chunks:
        if any_search_errored:
            log.warning("Query %r found no chunks after search infrastructure errors", question)
            yield {
                "type": "token",
                "text": (
                    "Sorry, I ran into a problem reaching the search service and "
                    "couldn't complete this search. Please try again in a moment."
                ),
            }
        else:
            log.info("Query %r returned no results (policy_area=%r)", question, policy_area)
            yield {"type": "token", "text": "I couldn't find any relevant passages in the indexed transcripts."}
        yield {"type": "done", "sources": []}
        return

    yield {"type": "progress", "message": f"Found {len(chunks)} passages. Composing cited answer…"}

    collected: dict[int, dict] = {}
    for ev in _synthesize(question, chunks, policy_area=policy_area, history_messages=history_messages):
        if ev["type"] == "citation" and ev["index"] not in collected:
            collected[ev["index"]] = {
                "cited_text": ev["cited_text"],
                "source": ev["source"],
                "index": ev["index"],
            }  # "pos" is per-occurrence and intentionally not carried into `done.sources`
        yield ev

    sources = [collected[i] for i in sorted(collected)]
    log.info(
        "Query answered: %d chunk(s) gathered over %d round(s), %d citation(s)",
        len(chunks), rounds_used, len(sources),
    )
    yield {"type": "done", "sources": sources}
