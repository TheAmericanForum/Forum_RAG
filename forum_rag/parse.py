"""Parse a verbose AssemblyAI transcript JSON into a normalized Transcript."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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


def _derive(filename: str, source_file: str) -> tuple[str, Optional[str], Optional[str]]:
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

    return session, table, date


def parse_transcript(data: dict[str, Any], *, filename: str = "") -> Transcript:
    turns: list[Turn] = []
    for t in data.get("turns", []) or []:
        text = (t.get("text") or "").strip()
        turns.append(
            Turn(
                turn_index=int(t.get("turn_index", 0) or 0),
                speaker=str(t.get("speaker", "?")),
                start_ms=int(t.get("start_ms", 0) or 0),
                end_ms=int(t.get("end_ms", 0) or 0),
                text=text,
                review_flag=bool(t.get("review_flag", False)),
            )
        )

    fname = filename or Path(data.get("source_file", "")).name
    session, table, date = _derive(fname, data.get("source_file", ""))

    return Transcript(
        transcript_id=data.get("transcript_id") or fname or "unknown",
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
    p = Path(path)
    data = json.loads(p.read_text())
    return parse_transcript(data, filename=p.name)
