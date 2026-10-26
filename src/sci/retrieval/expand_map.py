from pathlib import Path
import json,time,hashlib,os
import numpy as np
import torch
import sci.retrieval.core as core

class ExpansionMapSession(core.Session):
    def _identity(self, state, sid):
        status = 'IN' if sid in set(state['ids']) else 'OUT'
        return f'source_id={sid} [{status}] {core.clean(self.graph.source(sid)["title"])}'

    def _routes(self, block):
        return [r for r in block['routes'] if r['score'] is not None and r['source_id'] != block.get('anchor')][:10]

    def _path(self, state, nodes):
        text = 'source_id=' + nodes[0]
        for a,b in zip(nodes,nodes[1:]):
            entities = state['edges'][':'.join(sorted([a,b],key=int))]['shared_entities']
            text += ' — {' + ' / '.join(core.clean(x) for x in entities) + '} — source_id=' + b
        return text

    def _prepare_previews(self, state, delta):
        if os.environ.get('SCI_PREVIEW_MODE', 'lexical_top2') == 'lexical_top2':
            self._prepare_lexical_previews(state, delta)
            return
        searches = [b for b in delta['blocks'] if b['kind'] != 'E']
        if searches:
            self._prepare_search_segment_prefix_previews(state, dict(delta, blocks=searches))
        expansions = [b for b in delta['blocks'] if b['kind'] == 'E']
        if expansions:
            self._prepare_segment_prefix_previews(state, dict(delta, blocks=expansions))

    def _prepare_lexical_previews(self, state, delta):
        from sci.retrieval.lexical_preview import LexicalPreview
        records = state.setdefault('focused_previews', {})
        cache = state.setdefault('focused_window_cache', {})
        svc = self.engine()
        if not hasattr(svc, 'lexical_preview'):
            bm25 = svc.backends.bm25.retriever
            svc.lexical_preview = LexicalPreview(bm25.vocab_dict, bm25.scores['indptr'],
                                                 len(svc.backends.bm25.source_ids))
        for block in delta['blocks']:
            begin = time.perf_counter()
            query = block['query'].strip()
            expanding = block['kind'] == 'E'
            size = 600 if expanding else 240
            selection = f'top2_indexed_segments_lexical_bm25_window{size}_no_window_embedding'
            anchor = block.get('anchor') if expanding else None
            ids = list(dict.fromkeys(([anchor] if anchor else []) + [r['source_id'] for r in self._routes(block)]))
            query_calls = 0
            if expanding:
                assert query in svc.contexts, 'Expansion context must exist before preview'
                segment_scores = svc.contexts[query]['segment']
            else:
                segment_scores = svc.r.segment_score_cache.get(query)
            vector = None
            if segment_scores is None:
                if query not in svc.preview_vectors:
                    svc.preview_vectors[query] = svc.r.searcher.encoder.segment_queries([query],512)[0]
                    query_calls = 1
                vector = np.asarray(svc.preview_vectors[query], dtype=np.float32)
            for sid in ids:
                key = core.digest([selection, 'expand' if expanding else 'search', sid, query, size])
                record_key = str(block['revision']) + ':' + block['query_id'] + ':' + sid
                if key in cache:
                    records[record_key] = dict(cache[key], was_acquired_before=sid not in set(delta['new_ids']))
                    continue
                rows = svc.backends.source_segments[self.graph.lookup[sid]]
                if not len(rows):
                    records[record_key] = dict(source_id=sid, unavailable='No indexed segment')
                    continue
                if segment_scores is not None:
                    scores = np.asarray(segment_scores[rows])
                else:
                    with torch.inference_mode():
                        scores = svc.r.searcher.semantic_index.score_subset(vector, rows)
                chosen = sorted(range(len(rows)), key=lambda i: (-float(scores[i]), int(rows[i])))[:2]
                segments = []
                for i in chosen:
                    row = int(rows[i]); seq = int(svc.backends.segment_seqs[row])
                    assert seq == int(svc.r.searcher.semantic_index.unit_seqs[row])
                    meta = self.graph.db.execute('SELECT source_id,char_start,char_end FROM segment WHERE segment_seq=?', (seq,)).fetchone()
                    assert meta and str(meta[0]) == sid
                    segments.append(dict(start=int(meta[1]), end=int(meta[2]), seq=seq, row=row, score=float(scores[i])))
                body = (self.graph.corpus/self.graph.source(sid)['relpath']).read_bytes().decode('utf-8')
                lexical_begin = time.perf_counter()
                result = svc.lexical_preview.select(body, query, segments, size=size)
                lexical_seconds = time.perf_counter() - lexical_begin
                start, end, segment = result['start'], result['end'], result['segment']
                snippet = body[start:end]
                assert 0 < len(snippet) <= size
                record = dict(source_id=sid, query=query, chars=size, char_start=start, char_end=end,
                              charOffset=len(body[:start].encode('utf-16-le'))//2,
                              charLimit=len(snippet.encode('utf-16-le'))//2, snippet=snippet,
                              snippet_sha256=hashlib.sha256(snippet.encode()).hexdigest(),
                              selected_segment_seq=segment['seq'], selected_segment_row=segment['row'],
                              selected_window_position=start-segment['start'], selected_score=segment['score'],
                              indexed_segment_start=segment['start'], indexed_segment_end=segment['end'],
                              candidate_segments=segments, lexical_score=result['lexical_score'],
                              lexical_matched_terms=result['matched_terms'], lexical_fallback=result['fallback'],
                              lexical_candidate_count=result['candidate_count'], lexical_seconds=lexical_seconds,
                              selection=selection, tie_rule='lexical_bm25_then_dense_rank_then_character_offset',
                              no_labels=True, preview_query_embedding_calls=query_calls, encoded_window_count=0)
                cache[key] = record
                records[record_key] = dict(record, was_acquired_before=sid not in set(delta['new_ids']))
            state.setdefault('focused_preview_events', []).append(dict(revision=block['revision'],query_id=block['query_id'],
                query=query, anchor=anchor, shown_ids=ids, top10=[r['source_id'] for r in self._routes(block)],
                max_chars=size, encoded_window_count=0, preview_query_embedding_calls=query_calls,
                selection=selection, seconds=time.perf_counter()-begin))

    def _prepare_search_segment_prefix_previews(self, state, delta):
        records = state.setdefault('focused_previews', {})
        cache = state.setdefault('focused_window_cache', {})
        svc = self.engine()
        for block in delta['blocks']:
            begin = time.perf_counter()
            query, size = block['query'], 240
            ids = list(dict.fromkeys([r['source_id'] for r in self._routes(block)]))
            pending = []
            for sid in ids:
                key = core.digest(['search_segment_prefix', sid, query, size])
                record_key = str(block['revision']) + ':' + block['query_id'] + ':' + sid
                if key in cache:
                    records[record_key] = dict(cache[key], was_acquired_before=sid not in set(delta['new_ids']))
                else:
                    pending.append((sid,key,record_key))
            query_calls = 0
            segment_scores = svc.r.segment_score_cache.get(query)
            vector = None
            if pending and segment_scores is None:
                if query not in svc.preview_vectors:
                    svc.preview_vectors[query] = svc.r.searcher.encoder.segment_queries([query],512)[0]
                    query_calls = 1
                vector = np.asarray(svc.preview_vectors[query],dtype=np.float32)
            for sid,key,record_key in pending:
                rows = svc.backends.source_segments[self.graph.lookup[sid]]
                if not len(rows):
                    records[record_key] = dict(source_id=sid,unavailable='No indexed segment')
                    continue
                if segment_scores is not None:
                    scores = np.asarray(segment_scores[rows])
                else:
                    with torch.inference_mode():
                        scores = svc.r.searcher.semantic_index.score_subset(vector,rows)
                selected = min(range(len(rows)),key=lambda i:(-float(scores[i]),int(rows[i])))
                row = int(rows[selected])
                seq = int(svc.backends.segment_seqs[row])
                assert seq == int(svc.r.searcher.semantic_index.unit_seqs[row])
                meta = self.graph.db.execute('SELECT source_id,char_start,char_end FROM segment WHERE segment_seq=?',(seq,)).fetchone()
                assert meta and str(meta[0]) == sid
                body = (self.graph.corpus/self.graph.source(sid)['relpath']).read_bytes().decode('utf-8')
                start,segment_end = int(meta[1]),int(meta[2])
                assert 0 <= start < segment_end <= len(body)
                end = min(start+size,segment_end)
                snippet = body[start:end]
                record = dict(source_id=sid,query=query,chars=size,char_start=start,char_end=end,
                              charOffset=len(body[:start].encode('utf-16-le'))//2,
                              charLimit=len(snippet.encode('utf-16-le'))//2,snippet=snippet,
                              snippet_sha256=hashlib.sha256(snippet.encode()).hexdigest(),
                              selected_segment_seq=seq,selected_segment_row=row,
                              selected_window_position=0,selected_score=float(scores[selected]),
                              candidate_spans=[[start,end,seq,row]],candidate_scores=[float(scores[selected])],
                              indexed_segment_start=start,indexed_segment_end=segment_end,
                              selection='search_top1_existing_indexed_segment_prefix240_no_window_embedding',
                              tie_rule='first_indexed_segment_row',no_labels=True,encoded_window_count=0)
                cache[key] = record
                records[record_key] = dict(record,was_acquired_before=sid not in set(delta['new_ids']))
            state.setdefault('focused_preview_events',[]).append(dict(revision=block['revision'],query_id=block['query_id'],
                query=query,anchor=None,shown_ids=ids,top10=ids,max_chars=size,
                encoded_window_count=0,preview_query_embedding_calls=query_calls,
                selection='search_top1_existing_indexed_segment_prefix240_no_window_embedding',
                seconds=time.perf_counter()-begin))

    def _prepare_segment_prefix_previews(self, state, delta):
        records = state.setdefault('focused_previews', {})
        cache = state.setdefault('focused_window_cache', {})
        svc = self.engine()
        for block in delta['blocks']:
            begin = time.perf_counter()
            query, anchor = block['query'], block.get('anchor')
            size = 600
            # Reuse the segment scores computed by expand().
            context_key = query.strip()
            assert context_key in svc.contexts, 'Expansion context must exist before preview'
            segment_scores = svc.contexts[context_key]['segment']
            ids = list(dict.fromkeys(([anchor] if anchor else []) + [r['source_id'] for r in self._routes(block)]))
            for sid in ids:
                key = core.digest(['expand_segment_prefix', sid, context_key, size])
                record_key = str(block['revision']) + ':' + block['query_id'] + ':' + sid
                if key in cache:
                    records[record_key] = dict(cache[key], was_acquired_before=sid not in set(delta['new_ids']))
                    continue
                rows = svc.backends.source_segments[self.graph.lookup[sid]]
                if not len(rows):
                    records[record_key] = dict(source_id=sid, unavailable='No indexed segment')
                    continue
                scores = np.asarray(segment_scores[rows])
                selected = min(range(len(rows)), key=lambda i: (-float(scores[i]), i))
                row = int(rows[selected])
                seq = int(svc.backends.segment_seqs[row])
                assert seq == int(svc.r.searcher.semantic_index.unit_seqs[row])
                meta = self.graph.db.execute('SELECT source_id,char_start,char_end FROM segment WHERE segment_seq=?', (seq,)).fetchone()
                assert meta and str(meta[0]) == sid
                body = (self.graph.corpus/self.graph.source(sid)['relpath']).read_bytes().decode('utf-8')
                start, segment_end = int(meta[1]), int(meta[2])
                assert 0 <= start < segment_end <= len(body)
                end = min(start + size, segment_end)
                snippet = body[start:end]
                record = dict(source_id=sid, query=query, chars=size, char_start=start, char_end=end,
                              charOffset=len(body[:start].encode('utf-16-le'))//2,
                              charLimit=len(snippet.encode('utf-16-le'))//2, snippet=snippet,
                              snippet_sha256=hashlib.sha256(snippet.encode()).hexdigest(),
                              selected_segment_seq=seq, selected_segment_row=row,
                              selected_window_position=0, selected_score=float(scores[selected]),
                              candidate_spans=[[start,end,seq,row]], candidate_scores=[float(scores[selected])],
                              indexed_segment_start=start, indexed_segment_end=segment_end,
                              selection='top1_existing_indexed_segment_prefix_no_window_embedding',
                              tie_rule='first_indexed_segment_row', no_labels=True,
                              preview_query_embedding_calls=0, encoded_window_count=0)
                cache[key] = record
                records[record_key] = dict(record, was_acquired_before=sid not in set(delta['new_ids']))
            state.setdefault('focused_preview_events', []).append(dict(revision=block['revision'], query_id=block['query_id'],
                query=query, anchor=anchor, shown_ids=ids, top10=[r['source_id'] for r in self._routes(block)],
                max_chars=size, encoded_window_count=0, preview_query_embedding_calls=0,
                selection='top1_existing_indexed_segment_prefix_no_window_embedding', seconds=time.perf_counter()-begin))

    def _record(self,state,block,sid):
        return state.get('focused_previews',{}).get(str(block['revision'])+':'+block['query_id']+':'+sid)

    def _preview_lines(self,state,block,sid,full=False):
        record = self._record(state,block,sid)
        if not record:return []
        if 'unavailable' in record:return ['Passage unavailable: '+record['unavailable']]
        repeated = False
        if not full:
            current = state['blocks'].index(block)
            for previous in state['blocks'][:current]:
                old = self._record(state,previous,sid)
                if old and old.get('snippet_sha256') == record['snippet_sha256']:
                    repeated = True
                    break
        passage = 'Preview unchanged. The same raw window was shown previously.' if repeated else 'Preview: '+json.dumps(record['snippet'],ensure_ascii=False)
        return [passage, "Read this source:", f"path=.cep/s/{sid}.txt", f'charOffset={record["charOffset"]} charLimit={record["charLimit"]}']

    def _block_lines(self,state,block,newids=None,full=False):
        out = [('SEARCH' if block['kind']=='Q' else 'EXPANSION')+' '+block['query_id']+': '+block['query']]
        anchor = block.get('anchor')
        if anchor:
            out += ['──────── ANCHOR ────────', f'source_id={anchor} [{"IN" if anchor in set(state["ids"]) else "OUT"}]', 'Title: '+core.clean(self.graph.source(anchor)['title'])]
            out += self._preview_lines(state,block,anchor,full)
            out += ['───────────────────────', '']
        routes = self._routes(block)
        if not routes:out.append('No ranked related source.')
        for route in routes:
            sid = route['source_id']
            tag = ('NEW' if sid in newids else 'ALREADY ACQUIRED') if newids is not None else ''
            out += [f'──────── RESULT {route["rank"]} ────────', f'source_id={sid} [{"IN" if sid in set(state["ids"]) else "OUT"}]'+((' | '+tag) if tag else ''), 'Title: '+core.clean(self.graph.source(sid)['title'])]
            out.append('Path: '+self._path(state,route['path']))
            out += self._preview_lines(state,block,sid,full)
            out += ['─────────────────────────', '']
        intermediates = []
        for route in routes:intermediates.extend(route['path'][:-1])
        endpoint_ids = {r['source_id'] for r in routes}
        for sid in dict.fromkeys(intermediates):
            if sid != anchor and sid not in endpoint_ids:out.append('Via: '+self._identity(state,sid))
        return out

    def render_delta(self,state,delta):
        self._prepare_previews(state,delta)
        out = [f'CEP23 Map r{state["revision"]} | acquired={len(state["ids"])} | new={len(delta["new_ids"])}']
        for block in delta['blocks']:out += self._block_lines(state,block,set(delta['new_ids']))
        out += ['Acquired files: .cep/s/', 'Source catalogue: .cep/catalogue.tsv', 'Full Map and paths: .cep/MAP.txt']
        text = '\n'.join(out)+'\n'
        return dict(text=text,tokens=core.nt(text))

    def render_full(self,state):
        out = [f'CEP23 full Map r{state["revision"]}']
        for block in state['blocks']:out += self._block_lines(state,block,full=True)
        text = '\n'.join(out)+'\n'
        return dict(text=text,tokens=core.nt(text))

    def _write_views(self,state):
        super()._write_views(state)
