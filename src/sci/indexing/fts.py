import sqlite3
from sci.indexing.common import ASSETS, DATA, identity, progress, save


def build():
    path=ASSETS/'segment_bm25_fts5.sqlite'
    base=sqlite3.connect(f'file:{ASSETS/"base.sqlite"}?mode=ro',uri=True)
    if path.exists():path.unlink()
    db=sqlite3.connect(path)
    db.executescript('''PRAGMA journal_mode=WAL;
      CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
      CREATE VIRTUAL TABLE segment_fts USING fts5(text,tokenize='porter unicode61');''')
    completed=0
    for sid,rel in base.execute('SELECT source_id,relpath FROM source ORDER BY CAST(source_id AS INTEGER)'):
        text=(DATA/'documents'/rel).read_bytes().decode('utf-8')
        records=[(seq,text[a:b]) for seq,a,b in base.execute('SELECT segment_seq,char_start,char_end FROM segment WHERE source_id=? ORDER BY ordinal',(sid,))]
        db.executemany('INSERT INTO segment_fts(rowid,text) VALUES (?,?)',records)
        completed+=len(records)
        if int(sid)%1000==0:db.commit();progress('fts',rows=completed)
    db.executemany('INSERT INTO metadata VALUES (?,?)',[('protocol','sci-segment-fts5-bm25-v1'),('state','complete'),('build_identity',identity()),('segments',str(completed))])
    db.commit();db.execute("INSERT INTO segment_fts(segment_fts) VALUES ('optimize')");db.commit();db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();base.close()
    save(ASSETS/'FTS_READY.json',dict(status='PASS',rows=completed,build_identity=identity(),used_by_current_graphsearch=False))


if __name__=='__main__':build()
