import json
from bisect import bisect_right
from sci.indexing.segment_format import format_segment
import os
from pathlib import Path
import sqlite3
from concurrent.futures import ProcessPoolExecutor

from sci.indexing.common import ASSETS, DATA, MODEL, config, identity, progress, save, sha256

_TOKENIZER = None


def tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
        if not _TOKENIZER.is_fast:
            raise ValueError('Fast tokenizer with original-text offsets required')
    return _TOKENIZER


def split_text(text, tokenizer, max_tokens, overlap, *, title=None, heading_at=None):
    """Return raw spans; re-tokenize each boundary to enforce the input token limit."""
    if not text:
        return []
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets = encoded['offset_mapping']
    budget = max_tokens-tokenizer.num_special_tokens_to_add(False)
    def formatted(a,b):
        if title is None:return text[a:b]
        heading=heading_at(a) if heading_at else ""
        return format_segment(title,heading,text[a:b])
    if not 0 <= overlap < budget:
        raise ValueError('Overlap must be smaller than content token budget')
    if not offsets:
        count = len(tokenizer(formatted(0,len(text)), add_special_tokens=True)['input_ids'])
        if count > max_tokens:
            raise ValueError('Unexpected tokenizer empty-offset input')
        return [(0, len(text), count)]
    spans = []
    first = 0
    previous_start = -1
    while first < len(offsets):
        start = 0 if first == 0 else int(offsets[first][0])
        if start <= previous_start:
            first += 1
            continue
        available = budget
        if title is not None:
            heading=heading_at(start) if heading_at else ''
            overhead=len(tokenizer(format_segment(title,heading,''),add_special_tokens=True,truncation=False)['input_ids'])
            available=max_tokens-overhead
            if available<=overlap:
                raise ValueError('Title/Section overhead leaves no room for configured body overlap; no silent truncation')
        stop = min(len(offsets), first+available)
        while stop > first:
            end = len(text) if stop == len(offsets) else int(offsets[stop-1][1])
            count = len(tokenizer(formatted(start,end), add_special_tokens=True, truncation=False)['input_ids'])
            if count <= max_tokens:
                break
            stop -= 1
        if stop <= first or end <= start:
            raise ValueError('Cannot make a nonempty tokenizer-bounded raw span')
        spans.append((start, end, count))
        previous_start = start
        if end == len(text):
            break
        first = max(first+1, stop-overlap)
    assert spans and spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(a[1] >= b[0] for a, b in zip(spans, spans[1:])), 'Uncovered original characters'
    return spans


def process_source(row):
    cfg = config()
    path = DATA/'documents'/row['relpath']
    raw = path.read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('Corpus checksum mismatch: '+row['source_id'])
    text = raw.decode('utf-8')
    metadata=DATA/'headings.sqlite'
    changes=[]
    if cfg.get('segment_encoding_format')=='title-section-passage-v1' and not metadata.exists():
        raise FileNotFoundError('Heading metadata missing; run sci index prepare')
    if metadata.exists():
        with sqlite3.connect(f'file:{metadata}?mode=ro',uri=True) as db:
            changes=db.execute('SELECT char_start,heading FROM heading WHERE source_id=? ORDER BY char_start',(row['source_id'],)).fetchall()
    starts=[h[0] for h in changes]
    def heading_at(start):
        i=bisect_right(starts,start)-1
        return changes[i][1] if i>=0 else ''
    spans=split_text(text,tokenizer(),cfg['segment_max_tokens'],cfg['segment_overlap_tokens'],title=row['title'],heading_at=heading_at)
    return row,len(text),[(a,b,n,heading_at(a)) for a,b,n in spans]


def build():
    cfg = config()
    ASSETS.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(ASSETS/'base.sqlite')
    db.executescript('''
      PRAGMA journal_mode=WAL;
      CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS source(source_id TEXT PRIMARY KEY,relpath TEXT NOT NULL UNIQUE,title TEXT NOT NULL,
        bytes INTEGER NOT NULL,segment_count INTEGER NOT NULL,atom_count INTEGER NOT NULL,char_count INTEGER NOT NULL,sha256 TEXT NOT NULL) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS segment(segment_seq INTEGER PRIMARY KEY,segment_id TEXT NOT NULL UNIQUE,
        source_id TEXT NOT NULL,ordinal INTEGER NOT NULL,char_start INTEGER NOT NULL,char_end INTEGER NOT NULL,
        token_count INTEGER NOT NULL,heading TEXT NOT NULL,UNIQUE(source_id,ordinal));
    ''')
    prior = dict(db.execute('SELECT key,value FROM metadata'))
    if prior and prior.get('build_identity') != identity():
        raise ValueError('Refusing to resume a different segmentation configuration')
    db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('build_identity', identity()))
    db.commit()
    rows = [json.loads(line) for line in (DATA/'sources.jsonl').read_text().splitlines()]
    done = {row[0] for row in db.execute('SELECT source_id FROM source')}
    pending = [row for row in rows if row['source_id'] not in done]
    seq = db.execute('SELECT COALESCE(MAX(segment_seq),0) FROM segment').fetchone()[0]
    with ProcessPoolExecutor(max_workers=cfg['cpu_workers']) as pool:
        def results():
            for first in range(0,len(pending),200):
                yield from pool.map(process_source,pending[first:first+200],chunksize=4)
        for index, (row, chars, spans) in enumerate(results(), 1):
            sid = row['source_id']
            db.execute('INSERT INTO source VALUES (?,?,?,?,?,?,?,?)', (sid,row['relpath'],row['title'],row['bytes'],len(spans),0,chars,row['sha256']))
            records = []
            for ordinal, (start, end, count, heading) in enumerate(spans):
                seq += 1
                records.append((seq,f'{sid}:{ordinal}:{start}:{end}',sid,ordinal,start,end,count,heading))
            db.executemany('INSERT INTO segment VALUES (?,?,?,?,?,?,?,?)', records)
            if index % 200 == 0:
                db.commit()
                progress('segments', completed_sources=len(done)+index, total_sources=len(rows), segments=seq)
    db.commit()
    actual = db.execute('SELECT COUNT(*) FROM source').fetchone()[0]
    assert actual == len(rows) == cfg['source_count']
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    result = dict(status='PASS', build_identity=identity(), sources=actual, segments=seq,
                  tokenizer='Qwen3-Embedding-0.6B',max_tokens=cfg['segment_max_tokens'], overlap=cfg['segment_overlap_tokens'],
                  tokens_include_special=True, encoding_format="title-section-passage-v1", tokens_include_title_section_passage=True, raw_span_offsets_preserved=True, corpus_sha256=sha256(DATA/'sources.jsonl'))
    save(ASSETS/'SEGMENTS_READY.json',result)
    progress('segments', **result)


if __name__ == '__main__':
    build()
