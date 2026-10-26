import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
import sci.indexing.segment as segment
import sci.indexing.entities as entities
import sci.indexing.graph as graph


class Reproject(unittest.TestCase):
    def test_complete_source_filter_new_spans_df_and_transition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data=root/'data';assets=root/'assets'
            (data/'documents').mkdir(parents=True);(data/'ner_cache').mkdir()
            rows=[]
            for sid,text in [('0','X shared entity. '*80),('1','Second X shared entity. '*80),('2','Z unique entity.')]:
                raw=text.encode();rel=sid+'.txt';(data/'documents'/rel).write_bytes(raw)
                rows.append(dict(source_id=sid,relpath=rel,title=sid,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            (data/'sources.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            cfg={'cpu_workers':1,'source_count':3,'segment_max_tokens':512,'segment_overlap_tokens':64,'df_min':2,'df_max':4}
            with patch.object(segment,'DATA',data),patch.object(segment,'ASSETS',assets),patch.object(segment,'config',return_value=cfg),patch.object(segment,'identity',return_value='fixture'),patch.object(segment,'progress'):segment.build()
            cache=data/'ner_cache/cache.sqlite'
            with sqlite3.connect(cache) as db:
                db.executescript('''CREATE TABLE entity(entity_seq INTEGER PRIMARY KEY,entity_id TEXT,label TEXT,norm TEXT,display_name TEXT);
                  CREATE TABLE occurrence(occurrence_seq INTEGER PRIMARY KEY,entity_seq INTEGER,source_id TEXT,char_start INTEGER,char_end INTEGER,score REAL,surface TEXT);
                  CREATE TABLE processed_source(source_id TEXT PRIMARY KEY);''')
                db.executemany('INSERT INTO entity VALUES (?,?,?,?,?)',[(1,'x','organization','x','X'),(2,'z','organization','z','Z')])
                db.executemany('INSERT INTO processed_source VALUES (?)',[(str(i),) for i in range(3)])
                db.executemany('INSERT INTO occurrence VALUES (?,?,?,?,?,?,?)',[(1,1,'0',0,1,.9,'X'),(2,1,'1',7,8,.8,'X'),(3,2,'2',0,1,.9,'Z'),(4,1,'unfinished',0,1,.99,'X')])
            (data/'ner_cache/MANIFEST.json').write_text(json.dumps({'source_snapshots':{'cache.sqlite':{}}}))
            with patch.object(entities,'DATA',data),patch.object(entities,'ASSETS',assets),patch.object(entities,'config',return_value=cfg),patch.object(entities,'identity',return_value='fixture'),patch.object(entities,'progress'):entities.build()
            with sqlite3.connect(assets/'rare_entities.sqlite') as db:
                self.assertEqual(db.execute('SELECT source_df,segment_df,occurrence_count FROM entity').fetchall(),[(2,2,2)])
                self.assertEqual(db.execute('SELECT COUNT(*) FROM occurrence_origin').fetchone()[0],2)
                seg=db.execute('SELECT segment_id FROM entity_segment ORDER BY source_id').fetchall()
                self.assertTrue(all(s[0].startswith(('0:','1:')) for s in seg))
            with patch.object(graph,'ASSETS',assets),patch.object(graph,'identity',return_value='fixture'),patch.object(graph,'progress'):graph.build()
            matrix=sparse.load_npz(assets/'source_transition.npz').toarray()
            np.testing.assert_allclose(matrix,[[0,1,0],[1,0,0],[0,0,0]])
            self.assertEqual(json.loads((assets/'graph/source_ids.json').read_text()),['0','1','2'])


if __name__=='__main__':unittest.main()
