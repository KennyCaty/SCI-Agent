import hashlib
import json
import sqlite3
import sys
from sci.indexing.common import ROOT, DATA, config, save, sha256
from sci.indexing.ner_utils import infer_source, subword_safe_windows, stable_entity_id
from sci.retrieval.asset_config import ENTITY_LABELS


def build():
    from gliner2 import AutoExtractor
    cfg=json.loads((ROOT/'configs/indexing/ner.json').read_text())
    model_path=ROOT/'models/GLiNER2.5-small'
    if sha256(model_path/'model.safetensors')!=cfg['model_sha256']:
        raise ValueError('GLiNER checkpoint differs from configs/indexing/ner.json')
    cache=DATA/'ner_cache';cache.mkdir(exist_ok=True)
    db=sqlite3.connect(cache/'occurrences.sqlite')
    db.executescript('''CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);
      CREATE TABLE IF NOT EXISTS entity(entity_seq INTEGER PRIMARY KEY,entity_id TEXT UNIQUE,label TEXT,norm TEXT,display_name TEXT,max_score REAL,UNIQUE(label,norm));
      CREATE TABLE IF NOT EXISTS occurrence(occurrence_seq INTEGER PRIMARY KEY,entity_seq INTEGER,source_id TEXT,char_start INTEGER,char_end INTEGER,score REAL,surface TEXT);
      CREATE INDEX IF NOT EXISTS occurrence_source ON occurrence(source_id);
      CREATE INDEX IF NOT EXISTS occurrence_entity ON occurrence(entity_seq);
      CREATE TABLE IF NOT EXISTS processed_source(source_id TEXT PRIMARY KEY);''')
    signature=hashlib.sha256((json.dumps(cfg,sort_keys=True)+sha256(DATA/'sources.jsonl')).encode()).hexdigest()
    old=dict(db.execute('SELECT * FROM metadata'))
    if old and old.get('identity')!=signature:raise ValueError('NER inputs/config changed')
    db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('identity',signature));db.commit()
    model=AutoExtractor.from_pretrained(str(model_path),map_location=config()['device'],quantize=False,use_flashdeberta=False)
    done={r[0] for r in db.execute('SELECT source_id FROM processed_source')}
    for line in (DATA/'sources.jsonl').read_text().splitlines():
        row=json.loads(line);sid=row['source_id']
        if sid in done:continue
        text=(DATA/'documents'/row['relpath']).read_bytes().decode('utf-8')
        windows=subword_safe_windows(text,tokenizer=model.processor.tokenizer,word_size=cfg['window_words'],word_overlap=cfg['window_overlap_words'],maximum_subwords=cfg['maximum_subword_tokens'],subword_overlap=cfg['subword_overlap_tokens'])
        mentions,_=infer_source(model,text,labels=ENTITY_LABELS,threshold=cfg['threshold'],window_words=cfg['window_words'],overlap_words=cfg['window_overlap_words'],call_windows=cfg['inference_call_windows'],batch_size=cfg['inference_batch_size'],source_windows=windows)
        with db:
            for m in mentions:
                key=(m['label'],m['norm'])
                db.execute('INSERT INTO entity(entity_id,label,norm,display_name,max_score) VALUES (?,?,?,?,?) '
                           'ON CONFLICT(label,norm) DO UPDATE SET max_score=max(entity.max_score,excluded.max_score), '
                           'display_name=CASE WHEN excluded.max_score>entity.max_score THEN excluded.display_name ELSE entity.display_name END',
                           (stable_entity_id(*key),*key,m['surface'],m['score']))
                seq=db.execute('SELECT entity_seq FROM entity WHERE label=? AND norm=?',key).fetchone()[0]
                db.execute('INSERT INTO occurrence(entity_seq,source_id,char_start,char_end,score,surface) VALUES (?,?,?,?,?,?)',(seq,sid,m['char_start'],m['char_end'],m['score'],m['surface']))
            db.execute('INSERT INTO processed_source VALUES (?)',(sid,))
        done.add(sid)
        if len(done)%100==0:print(f'NER: {len(done)} sources',flush=True)
    db.close()
    save(cache/'MANIFEST.json',{'source_snapshots':{'occurrences.sqlite':{'sha256':sha256(cache/'occurrences.sqlite')}},'identity':signature})

if __name__=='__main__':build()
