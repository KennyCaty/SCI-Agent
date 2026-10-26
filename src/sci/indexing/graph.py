import json
import sqlite3
import sys

import numpy as np
from scipy import sparse

from sci.indexing.common import ASSETS, ROOT, identity, progress, save
from sci.retrieval.asset_config import FrozenConfig


def incidence(rows, columns, values, shape):
    values=np.asarray(values,np.float32)
    columns=np.asarray(columns,np.int32)
    totals=np.bincount(columns,weights=values,minlength=shape[1])
    weights=(values/totals[columns]).astype(np.float32)
    matrix=sparse.coo_matrix((weights,(rows,columns)),shape=shape).tocsr()
    matrix.sort_indices()
    return matrix


def build():
    base=sqlite3.connect(f'file:{ASSETS/"base.sqlite"}?mode=ro',uri=True)
    entities=sqlite3.connect(f'file:{ASSETS/"rare_entities.sqlite"}?mode=ro',uri=True)
    sources=list(base.execute('SELECT source_id,bytes FROM source ORDER BY CAST(source_id AS INTEGER),source_id'))
    segments=list(base.execute('SELECT segment_seq,segment_id,source_id FROM segment ORDER BY segment_seq'))
    eseqs=[r[0] for r in entities.execute('SELECT entity_seq FROM entity ORDER BY entity_seq')]
    source_lookup={r[0]:i for i,r in enumerate(sources)}
    segment_lookup={r[1]:i for i,r in enumerate(segments)}
    entity_lookup={seq:i for i,seq in enumerate(eseqs)}
    folder=ASSETS/'graph';folder.mkdir(exist_ok=True)
    si=[];sj=[];sv=[];di=[];dj=[];dv=[]
    for ent,seg,count,confidence in entities.execute('SELECT entity_seq,segment_id,occurrence_count,max_score FROM entity_segment'):
        si.append(segment_lookup[seg]);sj.append(entity_lookup[ent]);sv.append(np.log1p(count)*confidence)
    for ent,source,count,confidence in entities.execute('SELECT entity_seq,source_id,SUM(occurrence_count),MAX(max_score) FROM entity_segment GROUP BY entity_seq,source_id'):
        di.append(source_lookup[source]);dj.append(entity_lookup[ent]);dv.append(np.log1p(count)*confidence)
    sm=incidence(si,sj,sv,(len(segments),len(eseqs)))
    dm=incidence(di,dj,dv,(len(sources),len(eseqs)))
    sparse.save_npz(folder/'segment_entity.npz',sm)
    sparse.save_npz(folder/'source_entity.npz',dm)
    np.save(folder/'segment_seqs.npy',np.asarray([r[0] for r in segments],np.uint64))
    np.save(folder/'segment_sources.npy',np.asarray([source_lookup[r[2]] for r in segments],np.int32))
    np.save(folder/'entity_seqs.npy',np.asarray(eseqs,np.uint64))
    np.save(folder/'source_bytes.npy',np.asarray([r[1] for r in sources],np.uint64))
    ids=[r[0] for r in sources]
    save(folder/'source_ids.json',ids)
    weights=(dm @ dm.T).tocsr();weights.setdiag(0);weights.eliminate_zeros()
    degree=np.asarray(weights.sum(axis=1)).ravel()
    transition=(sparse.diags(1/np.maximum(degree,1e-30)) @ weights).T.tocsr();transition.sort_indices()
    sparse.save_npz(ASSETS/'source_transition.npz',transition)
    np.savez_compressed(ASSETS/'global.npz',source_ids=np.asarray(ids))
    save(folder/'manifest.json',dict(format='sci-cep-incidence-v1',state='complete',config_sha256=FrozenConfig(embedding_dimension=256).sha256,
        config_sha256_by_dimension={str(d):FrozenConfig(embedding_dimension=d).sha256 for d in (256,1024)},build_identity=identity(),sources=len(sources),segments=len(segments),entities=len(eseqs),source_entity_edges=dm.nnz,segment_entity_edges=sm.nnz))
    save(ASSETS/'transition_manifest.json',dict(state='complete',sources=len(sources),edges=transition.nnz,dangling_sources=int((degree==0).sum()),build_identity=identity()))
    base.close();entities.close()
    progress('graph',sources=len(sources),segments=len(segments),edges=transition.nnz,status='PASS')


if __name__ == '__main__':build()
