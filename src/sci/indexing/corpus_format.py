from __future__ import annotations

import re

from pathlib import Path

from urllib.parse import urlparse

TITLE_RE = re.compile(r"(?mi)^title:\s*(.+?)\s*$")

INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

WHITESPACE_RE = re.compile(r"\s+")

MAX_STEM_LEN = 140

def extract_title(text: str) -> str | None:
    match = TITLE_RE.search(text)
    if match:
        return match.group(1).strip()
    return None

def sanitize_name(value: str, fallback: str) -> str:
    value = INVALID_CHARS_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value).strip().strip(".")
    return value or fallback

def get_domain(url: str) -> str:
    hostname = urlparse(url).hostname or "unknown-domain"
    return sanitize_name(hostname.lower(), "unknown-domain")

def build_filename(title: str | None, url: str, docid: str) -> str:
    parsed = urlparse(url)
    path_name = Path(parsed.path).name
    fallback = path_name or f"doc-{docid}"
    stem = title or fallback
    stem = sanitize_name(stem, f"doc-{docid}")
    if len(stem) > MAX_STEM_LEN:
        stem = stem[:MAX_STEM_LEN].rstrip(" .")
    if not stem:
        stem = f"doc-{docid}"
    return f"{stem}.txt"

def unique_path(path: Path, docid: str, text: str) -> Path:
    if not path.exists():
        return path
    try:
        if path.read_text(encoding="utf-8") == text:
            return path
    except OSError:
        pass
    stem = path.stem
    suffix = path.suffix
    candidate = path.with_name(f"{stem}__docid_{docid}{suffix}")
    if not candidate.exists():
        return candidate
    try:
        if candidate.read_text(encoding="utf-8") == text:
            return candidate
    except OSError:
        pass
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}__docid_{docid}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        try:
            if candidate.read_text(encoding="utf-8") == text:
                return candidate
        except OSError:
            pass
        counter += 1

def source_title(relpath: str) -> str:
    return Path(relpath).stem.replace("__docid_", " [duplicate ")[:240]
