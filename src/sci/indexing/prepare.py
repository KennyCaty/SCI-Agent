"""Prepare corpus files, headings and BM25 from JSONL."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from sci.indexing.common import DATA, INDEX, CONFIG, config, save
from sci.indexing.heading_units import structured_segments
from sci.indexing.corpus_format import extract_title, get_domain, build_filename, unique_path, source_title


def prepare(path, *, trec=False):
    if (DATA/'sources.jsonl').exists():
        raise FileExistsError('Prepared sources already exist; use a fresh checkout for changed inputs')
    DATA.mkdir(parents=True,exist_ok=True)
    documents=DATA/'documents'; documents.mkdir(exist_ok=True)
    records=[]; texts=[]; seen=set(); official_map={}; official_seen=set()
    headings=sqlite3.connect(DATA/'headings.sqlite')
    headings.execute('CREATE TABLE IF NOT EXISTS heading(source_id TEXT,char_start INTEGER,heading TEXT,PRIMARY KEY(source_id,char_start))')
    headings.execute('DELETE FROM heading')
    with Path(path).open(encoding='utf-8') as stream:
        for ordinal,line in enumerate(stream):
            row=json.loads(line)
            sid=str(ordinal) if trec else str(row.get('source_id',row.get('docid','')))
            if trec:
                official=str(row['_id'])
                if not __import__('re').fullmatch(r'[A-Za-z0-9_-]+',official) or official in official_seen:raise ValueError('Unsafe or duplicate BEIR ID')
                official_map[sid]=official;official_seen.add(official)
                row={**row,'text':(row.get('title') or '')+'\n\n'+(row.get('text') or '')}
            if not sid or sid in seen or not sid.isdecimal():
                raise ValueError('Expected distinct numeric source_id/docid strings')
            seen.add(sid); text=row['text']
            if not isinstance(text,str):raise ValueError('Non-text document: '+sid)
            if trec:
                rel=official+'.txt'; title=source_title(rel);target=documents/rel
            else:
                url=str(row.get('url', ''))
                original_title=row.get('title') or extract_title(text)
                target_dir=documents/get_domain(url);target_dir.mkdir(parents=True,exist_ok=True)
                target=unique_path(target_dir/build_filename(original_title,url,sid),sid,text)
                rel=target.relative_to(documents).as_posix();title=source_title(rel)
            raw=text.encode('utf-8');target.write_bytes(raw)
            records.append(dict(source_id=sid,relpath=rel,title=title,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            # Use lexical boundaries for headings; dense segments use the Qwen tokenizer.
            last_heading=None
            for unit in structured_segments(text,sid,target_tokens=640,max_tokens=768):
                first=next((' '.join(x.split()) for x in unit.text.splitlines() if x.strip()),'')
                heading=first if len(first)<=180 and len(first.split())<=24 else ''
                if heading!=last_heading:
                    headings.execute('INSERT INTO heading VALUES (?,?,?)',(sid,unit.char_start,heading))
                    last_heading=heading
            texts.append(text)
    if len(records)!=config()['source_count']:
        raise ValueError(f"Expected {config()['source_count']} sources, got {len(records)}; check source_count in the selected indexing configuration")
    headings.commit();headings.close()
    if trec:save(DATA/'source_id_to_document_id.json',official_map)
    ordered=sorted(zip(records,texts),key=lambda item:(int(item[0]['source_id']),item[0]['source_id']))
    records=[r for r,t in ordered];texts=[t for r,t in ordered]
    import bm25s
    import Stemmer
    folder=INDEX/'source_bm25';folder.mkdir(parents=True,exist_ok=True)
    tokens=bm25s.tokenize(texts,stopwords='en',stemmer=Stemmer.Stemmer('english'))
    bm=bm25s.BM25(k1=1.5,b=.75,method='lucene',idf_method='lucene')
    bm.index(tokens);bm.save(str(folder/'bm25_index'))
    save(folder/'doc_ids.json',[r['source_id'] for r in records])
    records.sort(key=lambda r: (int(r['source_id']),r['source_id']))
    (DATA/'sources.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    print(f'Prepared {len(records)} documents, headings and BM25')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpus_jsonl',type=Path);p.add_argument('--trec-covid',action='store_true')
    args=p.parse_args()
    if args.trec_covid != (config().get('dataset')=='trec-covid'):p.error('Use SCI_INDEX_CONFIG=configs/indexing/trec_covid.json with --trec-covid')
    prepare(args.corpus_jsonl,trec=args.trec_covid)
