from __future__ import annotations

import hashlib, json, math, re, unicodedata

from typing import Any, Iterable, Mapping, Sequence

WORD_RE = re.compile(r"\S+")

SPACE_RE = re.compile(r"\s+")

def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def stable_entity_id(label: str, norm: str) -> str:
    raw = f"{label}\0{norm}".encode("utf-8", errors="replace")
    return "E" + hashlib.sha1(raw).hexdigest()

def normalize_surface(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    value = value.replace("\u2013", "-").replace("\u2014", "-")
    return SPACE_RE.sub(" ", value).strip(" \t\r\n.,;:!?()[]{}<>\"'").casefold()

def acceptable(surface: str, norm: str) -> bool:
    if not norm or len(norm) > 180 or not any(char.isalpha() for char in norm):
        return False
    if len(norm) == 1 and norm.isascii():
        return False
    return "\x00" not in surface

def windows(text: str, size: int, overlap: int) -> list[tuple[int, str]]:
    spans = [(match.start(), match.end()) for match in WORD_RE.finditer(text)]
    if not spans:
        return []
    step = size - overlap
    output: list[tuple[int, str]] = []
    for begin in range(0, len(spans), step):
        finish = min(begin + size, len(spans))
        char_start = spans[begin][0]
        char_end = spans[finish - 1][1]
        output.append((char_start, text[char_start:char_end]))
        if finish == len(spans):
            break
    return output

def subword_safe_windows(
    text: str,
    *,
    tokenizer: Any,
    word_size: int,
    word_overlap: int,
    maximum_subwords: int,
    subword_overlap: int,
) -> list[tuple[int, str]]:
    """Split windows at tokenizer offsets when they exceed the model token limit."""

    if maximum_subwords <= subword_overlap or maximum_subwords <= 0:
        raise ValueError("subword cap must be positive and exceed overlap")
    output: list[tuple[int, str]] = []
    seen: set[tuple[int, int]] = set()
    for base_start, piece in windows(text, word_size, word_overlap):
        encoded = tokenizer(
            piece,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        offsets = [
            (int(begin), int(end))
            for begin, end in encoded["offset_mapping"]
            if int(end) > int(begin)
        ]
        if not offsets:
            continue
        if len(offsets) <= maximum_subwords:
            ranges = [(0, len(offsets))]
        else:
            ranges = []
            step = maximum_subwords - subword_overlap
            for begin in range(0, len(offsets), step):
                end = min(begin + maximum_subwords, len(offsets))
                ranges.append((begin, end))
                if end == len(offsets):
                    break
        for begin, end in ranges:
            local_start = offsets[begin][0]
            local_end = offsets[end - 1][1]
            source_start = base_start + local_start
            source_end = base_start + local_end
            key = (source_start, source_end)
            if key in seen:
                continue
            seen.add(key)
            output.append((source_start, text[source_start:source_end]))
    output.sort(key=lambda value: value[0])
    return output

def chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for begin in range(0, len(values), size):
        yield values[begin : begin + size]

def parse_result(
    result: Mapping[str, Any],
    *,
    window_text: str,
    window_start: int,
    threshold: float,
) -> list[dict[str, Any]]:
    root = result.get("entities", result)
    if not isinstance(root, Mapping):
        return []
    output: list[dict[str, Any]] = []
    for raw_label, values in root.items():
        label = SPACE_RE.sub(" ", str(raw_label)).strip().casefold()
        if not label or not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            try:
                local_start = int(value["start"])
                local_end = int(value["end"])
                score = float(value.get("confidence", value.get("score", 0.0)))
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(score) or score < threshold:
                continue
            if not (0 <= local_start < local_end <= len(window_text)):
                continue
            surface = window_text[local_start:local_end]
            norm = normalize_surface(surface)
            if not acceptable(surface, norm):
                continue
            output.append(
                {
                    "label": label,
                    "surface": surface,
                    "norm": norm,
                    "score": score,
                    "char_start": window_start + local_start,
                    "char_end": window_start + local_end,
                }
            )
    return output

def infer_source(
    model: Any,
    text: str,
    *,
    labels: Sequence[str],
    threshold: float,
    window_words: int,
    overlap_words: int,
    call_windows: int,
    batch_size: int,
    source_windows: Sequence[tuple[int, str]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if source_windows is None:
        source_windows = windows(text, window_words, overlap_words)
    deduplicated: dict[tuple[str, int, int], dict[str, Any]] = {}
    for group in chunks(source_windows, call_windows):
        outputs = model.batch_extract_entities(
            [value[1] for value in group],
            list(labels),
            batch_size=batch_size,
            threshold=threshold,
            include_confidence=True,
            include_spans=True,
        )
        if len(outputs) != len(group):
            raise RuntimeError("GLiNER2 returned the wrong number of window results")
        for (window_start, window_text), result in zip(group, outputs):
            for mention in parse_result(
                result,
                window_text=window_text,
                window_start=window_start,
                threshold=threshold,
            ):
                key = (
                    str(mention["label"]),
                    int(mention["char_start"]),
                    int(mention["char_end"]),
                )
                previous = deduplicated.get(key)
                if previous is None or float(mention["score"]) > float(previous["score"]):
                    deduplicated[key] = mention
    return sorted(
        deduplicated.values(),
        key=lambda value: (value["char_start"], value["char_end"], value["label"]),
    ), len(source_windows)
