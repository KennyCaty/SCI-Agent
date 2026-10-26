import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
import sci.indexing.segment as segment


class BuildCorpus(unittest.TestCase):
    def test_build_and_resume_do_not_duplicate_or_renumber_segments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data=root/'data';assets=root/'assets'
            (data/'documents').mkdir(parents=True)
            rows=[]
            for sid,text in [('0','  Chinese 中文🙂.\r\n'+'Evidence statement. '*500),('1','Short second document.')]:
                raw=text.encode();rel=sid+'.txt';(data/'documents'/rel).write_bytes(raw)
                rows.append(dict(source_id=sid,relpath=rel,title=sid,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            (data/'sources.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            cfg={'cpu_workers':2,'source_count':2,'segment_max_tokens':1024,'segment_overlap_tokens':128}
            with patch.object(segment,'DATA',data),patch.object(segment,'ASSETS',assets),patch.object(segment,'config',return_value=cfg),patch.object(segment,'identity',return_value='fixture'),patch.object(segment,'progress'):
                segment.build()
                with sqlite3.connect(assets/'base.sqlite') as db:
                    first=list(db.execute('SELECT * FROM segment ORDER BY segment_seq'))
                    self.assertEqual(db.execute('SELECT COUNT(*) FROM source').fetchone()[0],2)
                    self.assertTrue(all(0<r[6]<=1024 for r in first))
                segment.build()
                with sqlite3.connect(assets/'base.sqlite') as db:self.assertEqual(first,list(db.execute('SELECT * FROM segment ORDER BY segment_seq')))
                with patch.object(segment,'identity',return_value='different'):
                    with self.assertRaises(ValueError):segment.build()


if __name__=='__main__':unittest.main()
