import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from sci.indexing.segment_format import format_segment
import numpy as np
from sci.indexing.common import ASSETS, DATA, MODEL, ROOT, config, identity, progress, save


def build():
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel,AutoTokenizer
    from sci.retrieval.asset_config import FrozenConfig
    cfg=config();device=torch.device(cfg['device'])
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for index construction')
    torch.cuda.set_device(device);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side='left')
    model=AutoModel.from_pretrained(MODEL,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').to(device).eval()
    model.config.use_cache=False
    base=sqlite3.connect(f'file:{ASSETS/"base.sqlite"}?mode=ro',uri=True)
    count=base.execute('SELECT COUNT(*) FROM segment').fetchone()[0]
    folders={dim:ASSETS/('segment_dense' if dim==256 else 'segment_dense1024') for dim in cfg['embedding_dimensions']}
    arrays={}
    checkpoint=ASSETS/'DENSE_CHECKPOINT.json'
    saved=json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    if saved and saved['build_identity']!=identity():raise ValueError('Dense checkpoint identity changed')
    completed=saved['rows'] if saved else 0
    for dim,folder in folders.items():
        folder.mkdir(exist_ok=True)
        target=folder/'embedding.f16'
        mode='r+' if target.exists() else 'w+'
        if completed and not target.exists():raise ValueError('Checkpoint missing dense data')
        arrays[dim]=np.memmap(target,dtype=np.float16,mode=mode,shape=(count,dim))
    seqs=np.asarray([r[0] for r in base.execute('SELECT segment_seq FROM segment ORDER BY segment_seq')],np.uint64)
    for folder in folders.values():
        ids=folder/'unit_seq.u64'
        if ids.exists():assert ids.read_bytes()==seqs.tobytes()
        else:ids.write_bytes(seqs.tobytes())
    rows=base.execute('''SELECT s.segment_seq,s.char_start,s.char_end,s.token_count,d.relpath,d.title,s.heading
         FROM segment s JOIN source d USING(source_id) ORDER BY s.segment_seq LIMIT -1 OFFSET ?''',(completed,))
    start_time=time.perf_counter();initial=completed;total_tokens=0;oom_count=0
    while block:=rows.fetchmany(cfg['embedding_bucket_rows']):
        texts=[];lengths=[];last_path=None;body=None
        for seq,left,right,n,rel,title,heading in block:
            if rel!=last_path:body=(DATA/'documents'/rel).read_bytes().decode('utf-8');last_path=rel
            texts.append(format_segment(title,heading,body[left:right]));lengths.append(n)
        order=sorted(range(len(block)),key=lambda i:(lengths[i],i))
        actual={dim:np.empty((len(block),dim),np.float16) for dim in folders}
        cursor=0;batch_size=cfg['embedding_batch_size']
        while cursor<len(order):
            selected=order[cursor:cursor+batch_size]
            tokens=tokenizer([texts[i] for i in selected],padding=True,truncation=False,return_tensors='pt')
            actual_lengths=tokens['attention_mask'].sum(dim=1).tolist()
            if actual_lengths!=[lengths[i] for i in selected] or max(actual_lengths)>cfg['segment_max_tokens']:
                raise ValueError('Segmentation/embedding tokenizer mismatch or input exceeds configured token bound')
            try:
                with torch.inference_mode():
                    hidden=model(**{k:v.to(device) for k,v in tokens.items()}).last_hidden_state
                    pooled=hidden[:,-1,:1024].float()
                    for dim in actual:
                        actual[dim][selected]=F.normalize(pooled[:,:dim],p=2,dim=1).cpu().numpy().astype(np.float16)
                total_tokens+=sum(actual_lengths);cursor+=len(selected)
            except torch.OutOfMemoryError:
                oom_count+=1
                if batch_size==1:raise
                batch_size=max(1,batch_size//2);torch.cuda.empty_cache()
                progress('dense',event='oom_batch_reduction',batch_size=batch_size,completed=completed)
        for dim,array in arrays.items():
            array[completed:completed+len(block)]=actual[dim];array.flush()
        completed+=len(block)
        save(checkpoint,dict(build_identity=identity(),rows=completed,total=count))
        elapsed=time.perf_counter()-start_time
        progress('dense',rows=completed,total_rows=count,rows_per_second=(completed-initial)/max(elapsed,1e-9),tokens_per_second=total_tokens/max(elapsed,1e-9),oom_batch_reductions=oom_count)
    for dim,folder in folders.items():
        save(folder/'manifest.json',dict(state='complete',kind='segment',dimension=dim,row_count=count,capacity=count,
          embedding_file='embedding.f16',id_file='unit_seq.u64',config_sha256=FrozenConfig(embedding_dimension=dim).sha256,
          build_identity=identity(),model='Qwen3-Embedding-0.6B',truncation=False,encoding_format='title-section-passage-v1'))
    save(ASSETS/'DENSE_READY.json',dict(status='PASS',rows=count,dimensions=cfg['embedding_dimensions'],build_identity=identity()))


if __name__=='__main__':build()
