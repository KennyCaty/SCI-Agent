import contextlib
import json
import os
from pathlib import Path
import sys
import time
import queue
import threading

os.environ['CEP_RUN_ROOT'] = str(Path(sys.argv[1]).resolve())
os.environ.update(CEP_EMBEDDING_ATTENTION='sdpa', OMP_NUM_THREADS='4',
                  OPENBLAS_NUM_THREADS='4', NUMBA_NUM_THREADS='4',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
protocol = sys.stdout
with contextlib.redirect_stdout(sys.stderr):
    from sci.runtime.embedding_config import configure_embedding
    config_file = os.environ.get('CEP_PI_EMBEDDING_CONFIG')
    embedding_options = json.loads(Path(config_file).read_text()) if config_file else {}
    embedding = configure_embedding(embedding_options)
    from sci.retrieval.session import LocalSession
    SessionClass = LocalSession
    import sci.retrieval.core as core
    import torch
    torch.set_num_threads(4)
    from sci.retrieval.retrieval_service import Service
    service = Service()
    if embedding.get('expected_source_map'):
        expected_mapping = json.loads(Path(embedding['expected_source_map']).read_text())
        if service.backends.mapping != expected_mapping:
            raise ValueError('Actual source database differs from evaluation mapping')
    searcher = service.r.searcher
    import numpy as np
    matrix = searcher.semantic_index.matrix
    if searcher.encoder.dimension != embedding['dimension'] or not matrix.is_cuda or tuple(matrix.shape) != (embedding['row_count'], embedding['dimension']):
        raise ValueError('Actual CUDA encoder/index shape disagrees with embedding profile')
    if not np.array_equal(searcher.semantic_index.unit_seqs, service.backends.segment_seqs):
        raise ValueError('Dense/graph segment row identities disagree')
    (Path(sys.argv[1]) / 'CUDA_BACKEND.json').write_text(json.dumps(dict(
        status='PASS', backend='cuda_exact_dense', matrix_shape=list(matrix.shape),
        graph_row_ids_verified=True, faiss_loaded=False, body_window_encoding=False,
        engine_root=str(Path(core.__file__).parent.resolve()),
        background_backend='cuda_csr_spmv', background_iterations=24)))
    vector = searcher.encoder.segment_queries(['Embedding dimension preflight'], 512)
    if vector.shape != (1, embedding['dimension']):
        raise ValueError('Actual query embedding shape disagrees with index')
    session = SessionClass(Path(sys.argv[1]), graph=core.GraphStore(), service=service)
    (Path(sys.argv[1]) / 'EMBEDDING_READY.json').write_text(json.dumps(embedding, indent=2) + '\n')

print(json.dumps({'ready': True, 'pid': os.getpid(), 'embedding': embedding}), file=protocol, flush=True)
requests = queue.Queue()
protocol_lock = threading.Lock()
def emit(value):
    with protocol_lock:
        print(json.dumps(value, ensure_ascii=False), file=protocol, flush=True)

def receive():
    # Read stdin concurrently so queued requests are timed separately from execution.
    try:
        for line in sys.stdin:
            request = json.loads(line)
            received = time.monotonic()
            emit(dict(event='received', id=request['id'], received_monotonic=received))
            requests.put((request, received))
    finally:
        requests.put(None)

threading.Thread(target=receive, name='cep-tool-receiver', daemon=True).start()
while True:
    item = requests.get()
    if item is None:
        break
    request, received = item
    start = time.monotonic()
    emit(dict(event='started', id=request['id'], received_monotonic=received,
              started_monotonic=start))
    try:
        if request['name'] not in ('graphsearch', 'graphexpand'):
            raise ValueError('Only graph tools are accepted by the bridge')
        with contextlib.redirect_stdout(sys.stderr):
            result = session.dispatch(request['name'], request['args'], request['id'])
        response = dict(id=request['id'], ok=True, result=result)
    except Exception as exc:
        response = dict(id=request['id'], ok=False, error=f'{type(exc).__name__}: {exc}')
    end = time.monotonic()
    response['timing'] = dict(received_monotonic=received, started_monotonic=start,
        ended_monotonic=end, queue_seconds=start-received, execution_seconds=end-start)
    emit(response)
