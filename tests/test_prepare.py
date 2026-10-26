import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import sci.indexing.prepare as prepare
from sci.indexing.ner_utils import infer_source

class Preparation(unittest.TestCase):
    def test_manifest_offsets_and_bm25(self):
        import bm25s
        import Stemmer
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw.jsonl'
            rows=[{'docid':'1','title':'First: Why / Evidence?','text':'Heading\n\nAlpha research evidence.\r\n中文🙂'},
                  {'docid':'2','title':'Second','text':'Another heading\n\nBeta telescope observation.'}]
            raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with patch.object(prepare,'DATA',root/'data'),patch.object(prepare,'INDEX',root/'index'),patch.object(prepare,'config',return_value={'source_count':2}):
                prepare.prepare(raw)
                with self.assertRaises(FileExistsError):prepare.prepare(raw)
            manifest=[json.loads(x) for x in (root/'data/sources.jsonl').read_text().splitlines()]
            self.assertEqual([r['source_id'] for r in manifest],['1','2'])
            self.assertEqual(manifest[0]['title'],'First Why Evidence')
            self.assertEqual((root/'data/documents'/manifest[0]['relpath']).read_bytes(),rows[0]['text'].encode())
            with sqlite3.connect(root/'data/headings.sqlite') as db:
                self.assertEqual(db.execute('SELECT heading FROM heading WHERE source_id="1" AND char_start=0').fetchone()[0],'Heading')
            bm=bm25s.BM25.load(str(root/'index/source_bm25/bm25_index'))
            tokens=bm25s.tokenize(['telescope'],stopwords='en',stemmer=Stemmer.Stemmer('english'),return_ids=False,show_progress=False)[0]
            self.assertGreater(bm.get_scores(tokens)[1],bm.get_scores(tokens)[0])

    def test_trec_export_preserves_original_document_identity_and_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'corpus.jsonl'
            rows=[{'_id':'paper-a','title':'Clinical study','text':'Original abstract.\n'},
                  {'_id':'paper-b','title':'Second title','text':'Original abstract.\n'}]
            raw.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with patch.object(prepare,'DATA',root/'data'),patch.object(prepare,'INDEX',root/'index'),patch.object(prepare,'config',return_value={'source_count':2}):
                prepare.prepare(raw,trec=True)
            self.assertEqual(json.loads((root/'data/source_id_to_document_id.json').read_text()),{'0':'paper-a','1':'paper-b'})
            self.assertEqual((root/'data/documents/paper-a.txt').read_bytes(),b'Clinical study\n\nOriginal abstract.\n')
            manifest=[json.loads(line) for line in (root/'data/sources.jsonl').read_text().splitlines()]
            self.assertEqual(manifest[0]['title'],'paper-a')
            self.assertEqual(manifest[1]['relpath'],'paper-b.txt')

    def test_ner_deduplicates_offsets_and_keeps_max_score(self):
        class Model:
            def batch_extract_entities(self,texts,labels,**kwargs):
                return [{'entities':{'organization':[{'start':0,'end':4,'confidence':score}]}} for score in [.6,.9]]
        mentions,count=infer_source(Model(),'Acme research',labels=['organization'],threshold=.5,window_words=384,overlap_words=64,call_windows=256,batch_size=2,source_windows=[(0,'Acme'),(0,'Acme')])
        self.assertEqual(count,2);self.assertEqual(len(mentions),1)
        self.assertEqual(mentions[0]['score'],.9);self.assertEqual(mentions[0]['norm'],'acme')

if __name__=='__main__':unittest.main()
