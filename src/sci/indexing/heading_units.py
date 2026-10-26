#!/usr/bin/env python3
"""Lexical text boundaries for corpus heading extraction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Iterable


# Count words, individual CJK characters and punctuation as lexical tokens.
LEXICAL_TOKEN_RE = re.compile(
    r"[A-Za-z\u00c0-\u024f0-9]+(?:['\u2019_-][A-Za-z\u00c0-\u024f0-9]+)*"
    r"|[\u3400-\u9fff]"
    r"|[^\W\d_]+"
    r"|[^\w\s]",
    flags=re.UNICODE,
)

BLANK_LINE_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")

SENTENCE_END_RE = re.compile(
    r"(?:[.!?]+(?:[\"'\u2019\u201d)\]]+)?(?=\s|$)|\n+)",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class LexicalToken:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class StructuredSegment:
    source_id: str
    segment_id: str
    ordinal: int
    char_start: int
    char_end: int
    token_count: int
    text: str

    @property
    def start(self) -> int:
        return self.char_start

    @property
    def end(self) -> int:
        return self.char_end

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    token_count: int


def lexical_tokens(text: str, *, offset: int = 0) -> list[LexicalToken]:
    return [
        LexicalToken(match.group(0), offset + match.start(), offset + match.end())
        for match in LEXICAL_TOKEN_RE.finditer(text)
    ]


def lexical_token_count(text: str) -> int:
    return sum(1 for _ in LEXICAL_TOKEN_RE.finditer(text))


def _validate_sizes(target_tokens: int, max_tokens: int, *, label: str) -> None:
    if target_tokens <= 0:
        raise ValueError(f"{label}_target_tokens must be positive")
    if max_tokens <= 0:
        raise ValueError(f"{label}_max_tokens must be positive")
    if target_tokens > max_tokens:
        raise ValueError(f"{label}_target_tokens cannot exceed {label}_max_tokens")


def _unit_id(prefix: str, source_id: str, start: int, end: int, text: str) -> str:
    payload = (
        f"{source_id}\0{start}\0{end}\0".encode("utf-8", errors="replace")
        + text.encode("utf-8", errors="replace")
    )
    return f"{prefix}_{hashlib.sha1(payload).hexdigest()[:16]}"


def _structural_blocks(text: str) -> list[_Span]:
    spans: list[_Span] = []
    cursor = 0
    for separator in BLANK_LINE_RE.finditer(text):
        spans.extend(_trimmed_nonempty_span(text, cursor, separator.start()))
        cursor = separator.end()
    spans.extend(_trimmed_nonempty_span(text, cursor, len(text)))
    return spans


def _trimmed_nonempty_span(text: str, start: int, end: int) -> list[_Span]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return []
    count = lexical_token_count(text[start:end])
    return [] if count == 0 else [_Span(start, end, count)]


def _sentence_spans(text: str, start: int, end: int) -> list[_Span]:
    boundaries = [start]
    for match in SENTENCE_END_RE.finditer(text, start, end):
        boundary = match.end()
        if start < boundary < end:
            boundaries.append(boundary)
    boundaries.append(end)

    spans: list[_Span] = []
    for left, right in zip(boundaries, boundaries[1:]):
        count = lexical_token_count(text[left:right])
        if count:
            spans.append(_Span(left, right, count))
        elif spans:
            previous = spans[-1]
            spans[-1] = _Span(previous.start, right, previous.token_count)
    return spans


def _hard_token_split(
    text: str,
    span: _Span,
    *,
    target_tokens: int,
) -> list[_Span]:
    tokens = lexical_tokens(text[span.start:span.end], offset=span.start)
    if not tokens:
        return []
    pieces: list[_Span] = []
    token_start = 0
    while token_start < len(tokens):
        token_end = min(len(tokens), token_start + target_tokens)
        char_start = span.start if token_start == 0 else tokens[token_start].start
        # Keep inter-token whitespace in the preceding span.
        char_end = span.end if token_end == len(tokens) else tokens[token_end].start
        pieces.append(_Span(char_start, char_end, token_end - token_start))
        token_start = token_end
    return pieces


def _pack_spans(
    spans: Iterable[_Span],
    *,
    target_tokens: int,
    max_tokens: int,
) -> list[_Span]:
    packed: list[_Span] = []
    current: _Span | None = None
    for span in spans:
        if current is None:
            current = span
            continue
        should_flush = (
            current.token_count >= target_tokens
            or current.token_count + span.token_count > max_tokens
        )
        if should_flush:
            packed.append(current)
            current = span
        else:
            current = _Span(current.start, span.end, current.token_count + span.token_count)
    if current is not None:
        packed.append(current)
    return packed


def _split_overlong_block(
    text: str,
    block: _Span,
    *,
    target_tokens: int,
    max_tokens: int,
) -> list[_Span]:
    if block.token_count <= max_tokens:
        return [block]

    safe_sentences: list[_Span] = []
    for sentence in _sentence_spans(text, block.start, block.end):
        if sentence.token_count <= max_tokens:
            safe_sentences.append(sentence)
        else:
            safe_sentences.extend(
                _hard_token_split(text, sentence, target_tokens=target_tokens)
            )
    return _pack_spans(
        safe_sentences,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
    )


def structured_segments(
    text: str,
    source_id: str,
    *,
    target_tokens: int = 640,
    max_tokens: int = 768,
) -> list[StructuredSegment]:
    """Partition the source into contiguous spans, preserving whitespace and offsets."""

    _validate_sizes(target_tokens, max_tokens, label="segment")
    blocks = _structural_blocks(text)
    if not blocks:
        return []

    units: list[_Span] = []
    for block in blocks:
        units.extend(
            _split_overlong_block(
                text,
                block,
                target_tokens=target_tokens,
                max_tokens=max_tokens,
            )
        )
    core = _pack_spans(units, target_tokens=target_tokens, max_tokens=max_tokens)

    # Include whitespace gaps so the spans cover the full source.
    covered: list[_Span] = []
    for index, span in enumerate(core):
        start = 0 if index == 0 else span.start
        end = len(text) if index == len(core) - 1 else core[index + 1].start
        count = lexical_token_count(text[start:end])
        if count > max_tokens:
            raise AssertionError("segment coverage normalization exceeded hard maximum")
        covered.append(_Span(start, end, count))

    segments: list[StructuredSegment] = []
    for ordinal, span in enumerate(covered):
        value = text[span.start:span.end]
        segments.append(
            StructuredSegment(
                source_id=source_id,
                segment_id=_unit_id("seg", source_id, span.start, span.end, value),
                ordinal=ordinal,
                char_start=span.start,
                char_end=span.end,
                token_count=span.token_count,
                text=value,
            )
        )
    return segments
