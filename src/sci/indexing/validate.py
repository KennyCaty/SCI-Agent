import json
import os
import sqlite3
import subprocess
import sys
import time

import numpy as np
from scipy import sparse
from sci.indexing.common import ASSETS, DATA, INDEX, MODEL, ROOT, config, identity, progress, save, sha256


def validate():
    inputs=json.loads((ROOT/'outputs/indexing/reports/BUILD_INPUTS.json').read_text())
    assert inputs['configuration']==config()
    for name,expected in inputs['file_sha256'].items():
        assert sha256(ROOT/name)==expected, 'Immutable build input changed: '+name
    cfg=config();base=sqlite3.connect(f'file:{ASSETS/"base.sqlite"}?mode=ro',uri=True)
    assert base.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert dict(base.execute('SELECT key,value FROM metadata'))['build_identity']==identity()
    for marker in ['SEGMENTS_READY.json','DENSE_READY.json','ENTITIES_READY.json','FTS_READY.json']:
        record=json.loads((ASSETS/marker).read_text())
        assert record['status']=='PASS' and record['build_identity']==identity()
    sources=list(base.execute('SELECT source_id,relpath,sha256,segment_count,char_count FROM source ORDER BY CAST(source_id AS INTEGER)'))
    segments=list(base.execute('SELECT segment_seq,source_id,ordinal,char_start,char_end,token_count FROM segment ORDER BY segment_seq'))
    assert len(sources)==cfg['source_count']
    for sid,rel,sha,count,chars in sources:
        assert sha256(DATA/'documents'/rel)==sha
        spans=list(base.execute('SELECT char_start,char_end,token_count FROM segment WHERE source_id=? ORDER BY ordinal',(sid,)))
        assert len(spans)==count
        if chars:assert spans[0][0]==0 and spans[-1][1]==chars and all(a[1]>=b[0] for a,b in zip(spans,spans[1:]))
        assert all(0<a[2]<=cfg['segment_max_tokens'] for a in spans)
    ids=[r[0] for r in sources]
    assert json.loads((ASSETS/'graph/source_ids.json').read_text())==ids
    assert set(map(str,json.loads((INDEX/'source_bm25/doc_ids.json').read_text())))==set(ids)
    seqs=np.asarray([r[0] for r in segments],np.uint64)
    np.testing.assert_array_equal(np.load(ASSETS/'graph/segment_seqs.npy'),seqs)
    dims={}
    for dim in cfg['embedding_dimensions']:
        folder=ASSETS/('segment_dense' if dim==256 else 'segment_dense1024')
        meta=json.loads((folder/'manifest.json').read_text())
        assert meta['state']=='complete' and meta['row_count']==len(segments) and meta['build_identity']==identity()
        assert meta['encoding_format']==cfg['segment_encoding_format']
        assert (folder/'embedding.f16').stat().st_size==len(segments)*dim*2
        np.testing.assert_array_equal(np.fromfile(folder/'unit_seq.u64',np.uint64),seqs)
        vectors=np.memmap(folder/'embedding.f16',dtype=np.float16,mode='r',shape=(len(segments),dim))
        worst=0.
        for i in range(0,len(vectors),8192):
            block=vectors[i:i+8192].astype(np.float32)
            assert np.isfinite(block).all()
            error=np.abs(np.linalg.norm(block,axis=1)-1)
            worst=max(worst,float(error.max()))
        assert worst<.002,worst
        dims[dim]={'rows':len(vectors),'max_unit_norm_error':worst}
    full=np.memmap(ASSETS/'segment_dense1024/embedding.f16',dtype=np.float16,mode='r',shape=(len(segments),1024))
    short=np.memmap(ASSETS/'segment_dense/embedding.f16',dtype=np.float16,mode='r',shape=(len(segments),256))
    for first in range(0,len(segments),8192):
        prefix=full[first:first+8192,:256].astype(np.float32)
        prefix/=np.linalg.norm(prefix,axis=1,keepdims=True)
        np.testing.assert_allclose(prefix,short[first:first+8192].astype(np.float32),atol=0.001,rtol=0.002)
    transition=sparse.load_npz(ASSETS/'source_transition.npz').tocsr()
    assert transition.shape==(len(ids),len(ids)) and np.isfinite(transition.data).all() and (transition.data>=0).all()
    sums=np.asarray(transition.sum(axis=0)).ravel()
    assert np.allclose(sums[sums>0],1,atol=2e-5)
    rare=sqlite3.connect(f'file:{ASSETS/"rare_entities.sqlite"}?mode=ro',uri=True)
    assert rare.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert rare.execute('SELECT COUNT(*) FROM entity WHERE source_df<? OR source_df>?',(cfg['df_min'],cfg['df_max'])).fetchone()[0]==0
    assert rare.execute('SELECT COUNT(*) FROM entity_segment s LEFT JOIN occurrence_origin o ON o.occurrence_seq=s.representative_occurrence_seq WHERE o.occurrence_seq IS NULL').fetchone()[0]==0
    env=dict(os.environ);env.pop('CEP_PI_DEPLOYMENT_CONFIG',None);env.pop('CEP_RESOURCES_CONFIG',None)
    for key in ('CEP_GRAPH_ASSETS','CEP_CORPUS','CEP_BM25_INDEX','CEP_EMBEDDING_MODEL'):env.pop(key,None)
    env['SCI_INDEX_VALIDATING']='1'
    from sci.indexing.common import DATASET
    env['CEP_RESOURCES_CONFIG']=str(ROOT/'configs/retrieval'/('trec_covid.json' if DATASET=='trec-covid' else 'browsecomp_plus.json'))
    node='node'
    check=subprocess.run([node,str(ROOT/'tests/retrieval_preflight.mjs')],capture_output=True,text=True,env=env,timeout=360)
    (ROOT/'outputs/indexing/logs/runtime-preflight.stdout.log').write_text(check.stdout)
    (ROOT/'outputs/indexing/logs/runtime-preflight.stderr.log').write_text(check.stderr)
    if check.returncode:raise RuntimeError('Real tool smoke failed: '+check.stderr[-3000:])
    smoke=json.loads(check.stdout);assert smoke['status']=='PASS'
    result=dict(status='PASS',at=time.time(),build_identity=identity(),sources=len(ids),segments=len(segments),dimensions=dims,
      tokenizer='Qwen3-Embedding-0.6B',segment_encoding_format=cfg['segment_encoding_format'],segment_max_tokens=cfg['segment_max_tokens'],segment_overlap_tokens=cfg['segment_overlap_tokens'],
      input_manifest_sha256=sha256(ROOT/'outputs/indexing/reports/BUILD_INPUTS.json'),
      raw_corpus_sha_verified=True,runtime=smoke)
    save(INDEX/'READY.json',result);save(ROOT/'outputs/indexing/reports/INDEX_REPORT.json',result)
    progress('complete',sources=len(ids),segments=len(segments),status='PASS')


if __name__=='__main__':validate()
