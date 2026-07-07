"""Parse a verbose AssemblyAI transcript JSON into a normalized Transcript."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Optional

log = logging.getLogger(__name__)

_TABLE_RE = re.compile(r"table\s*(\d+)", re.IGNORECASE)
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass
class Turn:
    turn_index: int
    speaker: str
    start_ms: int
    end_ms: int
    text: str
    review_flag: bool = False


@dataclass
class Transcript:
    transcript_id: str
    source_file: str
    source_sha256: str
    topic: str
    session: str
    table: Optional[str]
    date: Optional[str]
    speaker_count: int
    audio_duration_ms: int
    turns: list[Turn]


class DerivedMeta(NamedTuple):
    session: str
    table: Optional[str]
    date: Optional[str]


def _derive(filename: str, source_file: str) -> DerivedMeta:
    """Best-effort session/table/date from the filename and the source_file path."""
    session = ""
    if source_file:
        parts = [p for p in source_file.replace("\\", "/").split("/") if p]
        if len(parts) >= 2:
            session = parts[-2]  # the folder containing the recording
    if not session:
        session = Path(filename).stem

    table_m = _TABLE_RE.search(filename) or _TABLE_RE.search(source_file or "")
    table = table_m.group(1) if table_m else None

    date_m = _DATE_RE.search(filename) or _DATE_RE.search(source_file or "")
    date = date_m.group(1) if date_m else None

    return DerivedMeta(session, table, date)


def parse_transcript(data: dict[str, Any], *, filename: str = "") -> Transcript:
    """Normalize a raw AssemblyAI-style transcript dict into a Transcript.

    `filename` is the on-disk/Drive filename, used as a fallback whenever the JSON
    itself doesn't carry the needed metadata: if `filename` isn't passed, it falls
    back to the basename of `data["source_file"]`; `transcript_id` in turn falls back
    to that resolved filename, and finally to the literal string "unknown" if neither
    is available. Missing/non-numeric turn fields raise ValueError with the offending
    turn's index and this transcript's filename for context, since malformed input
    here would otherwise silently corrupt chunk boundaries and citation timestamps
    downstream.
    """
    filename = filename or Path(data.get("source_file", "")).name

    turns: list[Turn] = []
    for raw_turn in data.get("turns", []) or []:
        text = (raw_turn.get("text") or "").strip()
        try:
            turns.append(
                Turn(
                    turn_index=int(raw_turn.get("turn_index", 0) or 0),
                    speaker=str(raw_turn.get("speaker", "?")),
                    start_ms=int(raw_turn.get("start_ms", 0) or 0),
                    end_ms=int(raw_turn.get("end_ms", 0) or 0),
                    text=text,
                    review_flag=bool(raw_turn.get("review_flag", False)),
                )
            )
        except (TypeError, ValueError) as e:
            log.error("Malformed turn in %r: %s", filename or "<unknown file>", e)
            raise ValueError(
                f"Malformed turn (turn_index={raw_turn.get('turn_index')!r}) "
                f"in {filename or '<unknown file>'}: {e}"
            ) from e

    session, table, date = _derive(filename, data.get("source_file", ""))

    return Transcript(
        transcript_id=data.get("transcript_id") or filename or "unknown",
        source_file=data.get("source_file", ""),
        source_sha256=data.get("source_sha256", ""),
        topic=data.get("topic", ""),
        session=session,
        table=table,
        date=date,
        speaker_count=int(data.get("speaker_count", 0) or 0),
        audio_duration_ms=int(data.get("audio_duration_ms", 0) or 0),
        turns=turns,
    )


def load_transcript_file(path: str | Path) -> Transcript:
    """Read and parse a transcript JSON file from disk."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.error("Could not read/parse transcript file %s: %s", path, e)
        raise ValueError(f"Could not read/parse transcript file {path}: {e}") from e
    return parse_transcript(data, filename=path.name)
