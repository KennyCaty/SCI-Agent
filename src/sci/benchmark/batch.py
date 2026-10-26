"""Run QA inference on a JSONL question file."""
import argparse
import json
from pathlib import Path
import re
import sys
from sci.runtime.runner import run, runtime_args, runtime_options


def batch(config, questions, output, **options):
    rows = [json.loads(line) for line in Path(questions).read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        qid = str(row.get('id', row.get('query_id', '')))
        question = row.get('question', row.get('query'))
        if not re.fullmatch(r'[A-Za-z0-9_-]+', qid) or qid in seen:
            raise ValueError('Missing, unsafe or duplicate query ID')
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Empty question')
        seen.add(qid)
    root = Path(output).resolve()
    root.mkdir()
    (root / 'questions').mkdir()
    predictions = []
    for row in rows:
        qid = str(row.get('id', row.get('query_id')))
        question, folder = root / 'questions' / (qid + '.txt'), root / qid
        question.write_text(row.get('question', row.get('query')), encoding='utf-8')
        try:
            status = run(config, question, folder, **options)['status']
        except (OSError, ValueError, RuntimeError) as error:
            status = 'error'
            print(qid, str(error), file=sys.stderr)
        final = folder / 'final.txt'
        predictions.append(dict(query_id=qid, status=status,
                                prediction=final.read_text() if final.exists() else None))
        (root / 'predictions.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in predictions))
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'questions', 'output'):
        parser.add_argument(name)
    runtime_args(parser)
    args = parser.parse_args()
    predictions = batch(args.config, args.questions, args.output, **runtime_options(args))
    sys.exit(int(any(p['status'] != 'answered' for p in predictions)))
