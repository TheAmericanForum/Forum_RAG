"""Google Drive ingestion via a service account.

Credentials are loaded from the GOOGLE_SERVICE_ACCOUNT_JSON env var (the key file's
JSON contents) — no key file is written to disk, which suits Heroku's ephemeral FS.
"""
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from typing import Iterator, Optional

from .config import get_settings
from .errors import ConfigError, ExternalServiceError

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

_service = None


@dataclass
class DriveFile:
    id: str
    name: str
    modified_time: str
    md5: str


def _build_service():
    """Return the process-wide Drive API service client, creating it on first use."""
    global _service
    if _service is None:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        settings = get_settings()
        if not settings.google_service_account_json:
            raise ConfigError(
                "GOOGLE_SERVICE_ACCOUNT_JSON is not set. Paste the service-account key "
                "JSON into that env var (and share the Drive folders with its email)."
            )
        try:
            info = json.loads(settings.google_service_account_json)
        except json.JSONDecodeError as e:
            raise ConfigError(f"GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON: {e}") from e
        try:
            creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            _service = build("drive", "v3", credentials=creds, cache_discovery=False)
        except Exception as e:
            log.error("Failed to build Drive service: %s", e)
            raise ExternalServiceError(f"Could not authenticate with Google Drive: {e}") from e
    return _service


def _walk(service, folder_id: str) -> Iterator[DriveFile]:
    """Recursively yield every transcript JSON file under a Drive folder.

    Descends into subfolders via `yield from`, and pages through each folder's
    listing (`page_token`) until Drive stops returning a next-page token.
    """
    page_token: Optional[str] = None
    while True:
        try:
            resp = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed=false",
                    fields="nextPageToken, files(id,name,mimeType,modifiedTime,md5Checksum)",
                    pageToken=page_token,
                    pageSize=1000,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
        except Exception as e:
            log.error("Drive list failed for folder %r: %s", folder_id, e)
            raise ExternalServiceError(f"Google Drive list failed for folder {folder_id!r}: {e}") from e
        for drive_file in resp.get("files", []):
            mime = drive_file.get("mimeType")
            if mime == "application/vnd.google-apps.folder":
                yield from _walk(service, drive_file["id"])
            elif drive_file["name"].lower().endswith(".json") or mime == "application/json":
                yield DriveFile(
                    id=drive_file["id"],
                    name=drive_file["name"],
                    modified_time=drive_file.get("modifiedTime", ""),
                    md5=drive_file.get("md5Checksum", ""),
                )
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def list_transcript_files() -> list[DriveFile]:
    """List every transcript JSON file across all configured Drive folders."""
    settings = get_settings()
    if not settings.drive_folder_ids:
        raise ConfigError("DRIVE_FOLDER_IDS is not set (comma-separated Drive folder IDs).")
    service = _build_service()
    files: list[DriveFile] = []
    for folder_id in settings.drive_folder_ids:
        files.extend(_walk(service, folder_id))
    return files


def read_allowed_emails(file_id: str, tenant: str) -> list[str]:
    """Download a plain-text email allowlist from Drive and return emails for `tenant`.

    Format: sections headed by [tenant] (e.g. [sc], [nv], [nh]), one email per line.
    Lines starting with # and blank lines are ignored. A line that isn't a section
    header and appears before any section header belongs to no tenant and is skipped.
    """
    data = download_file(file_id)
    emails: list[str] = []
    current_section: str | None = None
    for line in data.decode(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].lower()
            continue
        if current_section == tenant.lower():
            emails.append(line.lower())
    return emails


def download_file(file_id: str) -> bytes:
    from googleapiclient.http import MediaIoBaseDownload

    service = _build_service()
    try:
        request = service.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    except Exception as e:
        log.error("Drive download failed for file_id=%r: %s", file_id, e)
        raise ExternalServiceError(f"Google Drive download failed for {file_id!r}: {e}") from e
    return buf.getvalue()
