"""Prepare, run and evaluate TREC-COVID."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from sci.runtime.runner import ROOT, runtime_args, runtime_options


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_qrels(file, query_ids, document_ids):
    lines = Path(file).read_text().strip().splitlines()
    if not lines or lines.pop(0).split() != ['query-id', 'corpus-id', 'score']:
        raise ValueError('Invalid BEIR qrels header')
    judgments = {}
    for line in lines:
        fields = line.split()
        if len(fields) != 3:
            raise ValueError('Invalid qrels row')
        q, d, value = fields
        grade = float(value)
        if q not in query_ids or d not in document_ids or not grade.is_integer() or not -1 <= grade <= 2:
            raise ValueError('Invalid qrel ID or grade')
        entry = judgments.setdefault(q, {})
        if d in entry:
            raise ValueError('Duplicate judgment')
        entry[d] = int(grade)
    for q in query_ids:
        if not any(g > 0 for g in judgments.get(q, {}).values()):
            raise ValueError('Missing positive qrels: ' + q)
    return judgments


def graded_metrics(ranking, qrels, k=10):
    ranked = list(dict.fromkeys(ranking))[:k]
    positive = sorted((g for g in qrels.values() if g > 0), reverse=True)
    def dcg(gains):
        return sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = dcg(positive[:k])
    hits = sum(qrels.get(d, 0) > 0 for d in ranked)
    return dict(ndcg_at_10=dcg([max(0, qrels.get(d, 0)) for d in ranked]) / ideal if ideal else 0,
                recall_at_10=hits / len(positive) if positive else 0, precision_at_10=hits / k,
                judged_at_10=sum(d in qrels and qrels[d] >= 0 for d in ranked) / k,
                unjudged_returned=sum(d not in qrels or qrels[d] < 0 for d in ranked))


def parse_ranking(text, mapping, excluded=(), workspace_paths=None):
    lines = text.replace('\\n', '\n').splitlines()
    start = None
    for i, line in enumerate(lines):
        plain = re.sub(r'^#{1,6}\s+', '', line.strip())
        plain = re.sub(r'\s+#+$', '', plain)
        plain = re.sub(r'\*\*|__', '', plain).strip()
        if re.fullmatch(r'Relevant Documents(?:\s*\(ranked\))?\s*:?\s*', plain, re.I):
            start = i + 1
            break
    if start is None:
        return dict(ranked=[], issues=['missing Relevant Documents block'])
    known, seen, blocked = set(mapping.values()), set(), set(excluded)
    path_ids = {path: sid for sid, path in (workspace_paths or {}).items()}
    ranked, issues = [], []
    for line in lines[start:]:
        match = re.fullmatch(r'\s*\d+[.)]\s+(.+?)\s*', line)
        if not match:
            continue
        token = match[1].replace('`', '').strip()
        sid = re.fullmatch(r'(?:source_id\s*=\s*|\.cep/s/)(\d+)(?:\.txt)?(?:\s+(?:[—–-]|:)\s+.*)?', token)
        sid = sid[1] if sid else token if re.fullmatch(r'\d+', token) else None
        doc = mapping.get(sid) if sid is not None else mapping.get(path_ids[token]) if token in path_ids else token
        if not doc or doc not in known:
            issues.append('unknown document: ' + token)
        elif doc in blocked:
            issues.append('excluded document: ' + str(doc))
        elif doc in seen:
            issues.append('duplicate document: ' + str(doc))
        else:
            seen.add(doc)
            ranked.append(doc)
            if len(ranked) == 10:
                break
    return dict(ranked=ranked, issues=issues)


def prepare(spec_file, data_dir, out):
    spec_file, data_dir, out = (Path(p).resolve() for p in (spec_file, data_dir, out))
    spec, base = load(spec_file), spec_file.parent
    assets = (base / spec['assets_root']).resolve()
    prepared = (base / spec['prepared_data']).resolve()
    profile = str(spec.get('embedding_profile', 256))
    if profile not in ('256', '1024'):
        raise ValueError('Invalid embedding profile')
    if load(assets / 'READY.json')['status'] != 'PASS':
        raise ValueError('Assets not ready')
    mapping = {str(r['source_id']): r['relpath'] for r in rows(prepared / 'sources.jsonl')}
    documents = {d['_id']: d for d in rows(data_dir / 'corpus.jsonl')}
    official = load(prepared / 'source_id_to_document_id.json')
    source_map = {}
    for sid in mapping:
        if not re.fullmatch(r'\d+', sid) or sid not in official or official[sid] not in documents:
            raise ValueError('Invalid source mapping')
        source_map[sid] = official[sid]
    if len(mapping) != len(documents) or len(set(source_map.values())) != len(documents):
        raise ValueError('Corpus mapping not bijective')
    queries = rows(data_dir / 'queries.jsonl')
    if len({q['_id'] for q in queries}) != len(queries) or any(
        not isinstance(q['_id'], str) or not re.fullmatch(r'[A-Za-z0-9_-]+', q['_id'])
        or not isinstance(q.get('text'), str) or not q['text'].strip() for q in queries
    ):
        raise ValueError('Invalid queries')
    qrels = read_qrels(data_dir / 'qrels/test.tsv', {q['_id'] for q in queries}, set(documents))
    if not spec.get('fixture') and (len(documents) != 171332 or len(queries) != 50):
        raise ValueError('Unexpected TREC-COVID dataset size')
    provider = load(base / spec['provider_config'])
    if provider.get('initial_full_workspace') or provider.get('workspace_naming') == 'id_title' or provider.get('tool_names'):
        raise ValueError('Use an unmodified four-tool provider config')
    resources = load(ROOT / 'configs/retrieval/browsecomp_plus.json')
    resources.update(graph_assets=str(assets / 'assets'), corpus=str(prepared / 'documents'),
                     bm25_index=str(assets / 'source_bm25'),
                     embedding_model=str((base / spec['embedding_model']).resolve()),
                     query_state=str(out / 'initial_state.json'))
    config = {**provider, 'task': dict(type='trec-covid', subset='trec-covid'), 'workspace_naming': 'id',
              'embedding': {**provider.get('embedding', {}), 'profile': profile, 'dimension': int(profile),
                            'query_max_length': 512,
                            **{k: resources[k] for k in ('graph_assets', 'corpus', 'bm25_index', 'embedding_model')},
                            'resources_config': str(out / 'resources.json'),
                            'source_map': str(out / 'source_corpus_map.json'),
                            'expected_source_map': str(out / 'source_corpus_map.json')}}
    prompt = (ROOT / 'prompts/trec_covid.txt').read_text()
    out.mkdir()
    (out / 'inputs').mkdir()
    (out / 'cases').mkdir()
    for name, value in [('source_map', source_map), ('source_corpus_map', mapping),
                        ('RUN_CONFIG', config), ('resources', resources),
                        ('initial_state', dict(gold_blind_generation=True,
                         initial_workspace_method='empty; query-driven native CEP tools', records=[]))]:
        save(out / (name + '.json'), value)
    (out / 'evaluation_input.jsonl').write_text(''.join(json.dumps(dict(id=q['_id'], query=q['text'], qrels=qrels[q['_id']])) + '\n' for q in queries))
    for q in queries:
        (out / 'inputs' / (q['_id'] + '.txt')).write_text('Retrieve and rank documents for this query.\n\nQUERY:\n' + q['text'])
    save(out / 'PLAN.json', dict(task='trec-covid', questions=len(queries), documents=len(documents),
         evaluation='BEIR test; linear graded nDCG@10; document IDs; unjudged gain=0, tracked separately; failures excluded and reported',
         query_field='text only', system_prompt=prompt,
         source_map_sha256=digest(prepared / 'source_id_to_document_id.json'),
         provider_config_sha256=digest(base / spec['provider_config']),
         input_hashes={name: digest(data_dir / name) for name in ('corpus.jsonl', 'queries.jsonl', 'qrels/test.tsv')},
         model_api_calls=0))
    return dict(out=str(out), questions=len(queries), documents=len(documents))


def evaluate(root):
    root = Path(root).resolve()
    inputs, mapping = rows(root / 'evaluation_input.jsonl'), load(root / 'source_map.json')
    results = []
    for row in inputs:
        folder = root / 'cases' / row['id']
        if not (folder / 'COMPLETE.json').exists():
            continue
        complete = load(folder / 'COMPLETE.json')
        answered = complete['status'] == 'answered'
        final = folder / 'final.txt'
        parsed = parse_ranking(final.read_text() if final.exists() else '', mapping)
        record = dict(id=row['id'], status=complete['status'], **parsed,
                      **(graded_metrics(parsed['ranked'], row['qrels']) if answered else dict(ndcg_at_10=None, recall_at_10=None)))
        for source, dest in [('decisions', 'turns'), ('tools', 'tools')]:
            if source in complete:
                record[dest] = complete[source]
        if (folder / 'TOKEN_USAGE.json').exists():
            usage = load(folder / 'TOKEN_USAGE.json')
            for source, dest in [('token_counts', 'token_counts'), ('total_cost', 'cost'), ('known_usage_cost', 'known_usage_cost'), ('currency', 'currency'), ('usage_complete', 'usage_complete')]:
                if source in usage:
                    record[dest] = usage[source]
        results.append(record)
    valid = [r for r in results if r['status'] == 'answered']
    report = dict(total=len(inputs), completed=len(results), answered=len(valid),
                  failed=len(results) - len(valid), pending=len(inputs) - len(results),
                  ndcg_at_10=sum(r['ndcg_at_10'] for r in valid) / len(valid) if valid else None,
                  recall_at_10=sum(r['recall_at_10'] for r in valid) / len(valid) if valid else None,
                  scope='answered only; format errors score zero; operational failures reported separately, never silently zero',
                  results=results)
    save(root / 'TREC_COVID_EVALUATION.json', report)
    return report


def launch(root, concurrency=1, **options):
    root = Path(root).resolve()
    if not 1 <= concurrency <= 32:
        raise ValueError('Concurrency must be 1..32')
    inputs = rows(root / 'evaluation_input.jsonl')
    if any((root / 'cases' / row['id']).exists() for row in inputs):
        raise FileExistsError('Existing attempt found; no implicit retries/resume')
    config = load(root / 'RUN_CONFIG.json')
    with (root / 'STARTED.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), concurrency=concurrency, started_at=datetime.now(timezone.utc).isoformat(),
                       api=options.get('api', config.get('api', 'openai-completions')),
                       minimum_request_interval_seconds=options.get('requestIntervalSeconds', config.get('minimum_request_interval_seconds', 0)),
                       request_interval_scope='per question; time from prior logical request completion to next request start',
                       runtime_overrides=options, run_config_sha256=digest(root / 'RUN_CONFIG.json')), stream)
    pending = iter(inputs)
    children = {}
    def stop(signum, _frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        exhausted = False
        while children or not exhausted:
            while not exhausted and len(children) < concurrency:
                row = next(pending, None)
                if row is None:
                    exhausted = True
                    break
                command = [sys.executable, '-m', 'sci', 'run', str(root / 'RUN_CONFIG.json'),
                           str(root / 'inputs' / (row['id'] + '.txt')), str(root / 'cases' / row['id'])]
                for key, flag in [('api', '--api'), ('requestIntervalSeconds', '--request-interval-seconds')]:
                    if key in options:
                        command += [flag, str(options[key])]
                children[subprocess.Popen(command, stdout=subprocess.DEVNULL)] = row
            for child, row in list(children.items()):
                if child.poll() is None:
                    continue
                del children[child]
                evaluate(root)
                folder = root / 'cases' / row['id']
                complete = folder / 'COMPLETE.json'
                status = load(complete)['status'] if complete.exists() else None
                if child.returncode or status not in ('answered', 'turn_limit', 'normal_time_limit') or any((folder / name).exists() for name in ('STORAGE_ERROR.json', 'ACCOUNTING_ERROR.json')):
                    raise RuntimeError(f"Question {row['id']} failed; dispatch stopped without retry")
            if children:
                time.sleep(.1)
        report = evaluate(root)
        if report['pending']:
            raise RuntimeError('Interrupted/incomplete ranking run')
        save(root / 'FINISHED.json', dict(completed=report['completed'], ndcg_at_10=report['ndcg_at_10']))
        return report
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prep = modes.add_parser('prepare')
    for name in ('spec', 'data', 'output'):
        prep.add_argument(name)
    ev = modes.add_parser('evaluate')
    ev.add_argument('root')
    launch_parser = modes.add_parser('run')
    launch_parser.add_argument('root')
    launch_parser.add_argument('--concurrency', type=int, default=1)
    runtime_args(launch_parser)
    args = parser.parse_args()
    if args.mode == 'prepare':
        result = prepare(args.spec, args.data, args.output)
    elif args.mode == 'evaluate':
        result = evaluate(args.root)
    else:
        result = launch(args.root, args.concurrency, **runtime_options(args))
    print(json.dumps(result, indent=2))
