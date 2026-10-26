import bisect
from collections import defaultdict
import json
import math
import sqlite3

from sci.indexing.common import ASSETS, DATA, config, identity, progress, save


def choose_segment(spans, start, end):
    """Maximum character overlap, then midpoint distance, then ordinal."""
    ends = [s[2] for s in spans]
    first = bisect.bisect_right(ends, start)
    candidates = []
    midpoint = (start+end)/2
    for ordinal, left, right, sid in spans[first:]:
        if left >= end:
            break
        overlap = max(0, min(end, right)-max(start, left))
        if overlap:
            candidates.append((-overlap,abs((left+right)/2-midpoint),ordinal,sid))
    if not candidates:
        raise ValueError(f'NER occurrence outside segment coverage: {start}:{end}')
    selected = min(candidates)
    return selected[2], selected[3]


def build():
    cfg = config()
    cache = DATA/'ner_cache'
    names = list(json.loads((cache/'MANIFEST.json').read_text())['source_snapshots'])
    source_rows = [json.loads(x) for x in (DATA/'sources.jsonl').read_text().splitlines()]
    expected_sources = {x['source_id'] for x in source_rows}
    seen = set()
    stats = {}
    local_maps = {}
    for name in names:
        db = sqlite3.connect(f'file:{cache/name}?mode=ro',uri=True)
        source_ids = {str(x[0]) for x in db.execute('SELECT source_id FROM processed_source')}
        assert not (source_ids & seen), 'NER source coverage overlaps'
        seen.update(source_ids)
        mapping = {}
        for seq,label,norm,display,eid,df,count,score in db.execute('''
            SELECT e.entity_seq,e.label,e.norm,e.display_name,e.entity_id,
              COUNT(DISTINCT o.source_id),COUNT(*),MAX(o.score)
            FROM occurrence o JOIN entity e USING(entity_seq) JOIN processed_source p USING(source_id) GROUP BY o.entity_seq'''):
            key = (label,norm)
            mapping[seq] = key
            if key not in stats:
                stats[key] = [display,eid,df,count,score]
            else:
                old=stats[key];old[2]+=df;old[3]+=count;old[4]=max(old[4],score)
        local_maps[name] = mapping
        db.close()
        progress('entities',event='global_df',cache=name,covered_sources=len(seen),entities=len(stats))
    assert seen == expected_sources, (len(seen),len(expected_sources))
    rare = {key: i+1 for i,key in enumerate(sorted(k for k,v in stats.items() if cfg['df_min'] <= v[2] <= cfg['df_max']))}
    output = ASSETS/'rare_entities.sqlite'
    if output.exists():
        output.unlink()
    result = sqlite3.connect(output)
    result.executescript('''
      CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
      CREATE TABLE entity(entity_seq INTEGER PRIMARY KEY,entity_id TEXT NOT NULL UNIQUE,label TEXT NOT NULL,
        norm TEXT NOT NULL,display_name TEXT NOT NULL,max_score REAL NOT NULL,source_df INTEGER NOT NULL,
        segment_df INTEGER NOT NULL,occurrence_count INTEGER NOT NULL,UNIQUE(label,norm));
      CREATE TABLE entity_segment(entity_seq INTEGER NOT NULL,segment_id TEXT NOT NULL,source_id TEXT NOT NULL,
        segment_ordinal INTEGER NOT NULL,representative_occurrence_seq INTEGER NOT NULL,occurrence_count INTEGER NOT NULL,
        max_score REAL NOT NULL,PRIMARY KEY(entity_seq,segment_id)) WITHOUT ROWID;
      CREATE TABLE occurrence_origin(occurrence_seq INTEGER PRIMARY KEY,cache_file TEXT NOT NULL,raw_occurrence_seq INTEGER NOT NULL,source_id TEXT NOT NULL,char_start INTEGER NOT NULL,char_end INTEGER NOT NULL);
    ''')
    for key,seq in rare.items():
        display,eid,df,count,score=stats[key]
        result.execute('INSERT INTO entity VALUES (?,?,?,?,?,?,?,?,?)',(seq,eid,*key,display,score,df,0,count))
    result.commit()
    base=sqlite3.connect(f'file:{ASSETS/"base.sqlite"}?mode=ro',uri=True)
    total=0
    for name in names:
        db=sqlite3.connect(f'file:{cache/name}?mode=ro',uri=True)
        allowed={local:rare[key] for local,key in local_maps[name].items() if key in rare}
        db.execute('CREATE TEMP TABLE selected(entity_seq INTEGER PRIMARY KEY)')
        db.executemany('INSERT INTO selected VALUES (?)',((x,) for x in allowed))
        cursor=db.execute('''SELECT o.source_id,o.entity_seq,o.char_start,o.char_end,o.score,o.occurrence_seq,o.surface
            FROM selected a JOIN occurrence o USING(entity_seq) JOIN processed_source p USING(source_id) ORDER BY o.source_id,o.char_start,o.char_end,o.entity_seq''')
        current=None;counts={};spans=None;body=None
        def flush():
            if current is None:return
            result.executemany('INSERT INTO entity_segment VALUES (?,?,?,?,?,?,?)',
                ((ent,seg,current,ordinal,rep,n,score) for (ent,seg),(ordinal,rep,n,score) in counts.items()))
        for sid,local,start,end,score,occ,surface in cursor:
            if sid != current:
                flush();current=sid;counts={}
                spans=list(base.execute('SELECT ordinal,char_start,char_end,segment_id FROM segment WHERE source_id=? ORDER BY ordinal',(sid,)))
                rel=base.execute('SELECT relpath FROM source WHERE source_id=?',(sid,)).fetchone()[0]
                body=(DATA/'documents'/rel).read_bytes().decode('utf-8')
            if not 0 <= start < end <= len(body) or body[start:end] != surface:
                raise ValueError(f'Raw NER occurrence/corpus mismatch {sid}:{start}:{end}')
            ordinal,segment_id=choose_segment(spans,start,end)
            representative=total+1
            result.execute('INSERT INTO occurrence_origin VALUES (?,?,?,?,?,?)',(representative,name,occ,sid,start,end))
            key=(allowed[local],segment_id)
            old=counts.get(key,(ordinal,representative,0,0.))
            counts[key]=(ordinal,old[1],old[2]+1,max(old[3],score))
            total+=1
        flush();db.close();result.commit()
        progress('entities',event='reproject',cache=name,rare_entities=len(rare),occurrences=total)
    result.execute('UPDATE entity SET segment_df=(SELECT COUNT(*) FROM entity_segment s WHERE s.entity_seq=entity.entity_seq)')
    mismatch=result.execute('''SELECT COUNT(*) FROM entity e WHERE e.source_df<>(SELECT COUNT(DISTINCT source_id) FROM entity_segment s WHERE s.entity_seq=e.entity_seq)
      OR e.occurrence_count<>(SELECT COALESCE(SUM(occurrence_count),0) FROM entity_segment s WHERE s.entity_seq=e.entity_seq)''').fetchone()[0]
    assert mismatch == 0,mismatch
    result.execute('INSERT INTO metadata VALUES (?,?)',('build_identity',identity()))
    result.commit();result.close();base.close()
    save(ASSETS/'ENTITIES_READY.json',dict(status='PASS',entities=len(rare),raw_occurrences_reprojected=total,complete_disjoint_sources=len(seen),build_identity=identity()))


if __name__ == '__main__':build()
