from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from sci.paths import ROOT


def frame(kind, **body):
    return 'event: ' + kind + '\ndata: ' + json.dumps(dict(type=kind, **body)) + '\n\n'


def response_stream(number, tool_call):
    if tool_call:
        item = dict(id=f'fc_{number}', type='function_call', call_id=f'call_{number}',
                    name='graphsearch', arguments=json.dumps({'queries': ['What is photosynthesis?']}))
    else:
        text = 'Explanation: synthetic fixture.\nExact Answer: ' + ('FIXTURE_GOLD_1' if number == 1 else 'FIXTURE_WRONG_2')
        item = dict(id=f'msg_{number}', type='message', role='assistant', status='completed',
                    content=[dict(type='output_text', text=text, annotations=[])])
    response = dict(id=f'resp_{number}_{int(tool_call)}', model='gpt-6-luna', status='completed',
                    usage=dict(input_tokens=100, input_tokens_details=dict(cached_tokens=0),
                               output_tokens=10, total_tokens=110), output=[item])
    stream = frame('response.created', response={**response, 'status': 'in_progress', 'usage': None, 'output': []})
    stream += frame('response.output_item.added', output_index=0,
                    item={**item, **({'arguments': ''} if tool_call else {'content': []})})
    if tool_call:
        stream += frame('response.function_call_arguments.delta', output_index=0, item_id=item['id'], delta=item['arguments'])
        stream += frame('response.function_call_arguments.done', output_index=0, item_id=item['id'], arguments=item['arguments'])
    else:
        stream += frame('response.content_part.added', output_index=0, content_index=0,
                        item_id=item['id'], part=dict(type='output_text', text='', annotations=[]))
        stream += frame('response.output_text.delta', output_index=0, content_index=0,
                        item_id=item['id'], delta=item['content'][0]['text'])
    return stream + frame('response.output_item.done', output_index=0, item=item) + frame('response.completed', response=response)


def main():
    generation, judging = [], []
    counts = {1: 0, 2: 0, 3: 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/judge/responses':
                judging.append(payload)
                prompt = payload['input'][-1]['content']
                correct = 'Exact Answer: FIXTURE_GOLD_1' in prompt
                body = json.dumps(dict(output_text=json.dumps(dict(is_correct=correct,
                    normalized_prediction='synthetic', reason='Deterministic test fixture')),
                    usage=dict(input_tokens=100, output_tokens=10))).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != '/generation/responses':
                self.send_error(404)
                return
            generation.append(payload)
            encoded = json.dumps(payload)
            number = next(n for n in counts if f'Synthetic question {n}:' in encoded)
            counts[number] += 1
            if number == 3:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"Deliberate fixture failure","type":"server_error"}}')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            self.wfile.write(response_stream(number, counts[number] == 1).encode())

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix='sci-bcp-preflight-') as tmp:
            folder = Path(tmp)
            questions = folder/'questions.jsonl'
            rows = [dict(id=f'fixture-{n}', question=f'Synthetic question {n}: What is photosynthesis?',
                         answer=f'FIXTURE_GOLD_{n}') for n in (1, 2, 3)]
            questions.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            config = json.loads((ROOT/'configs/models/gpt-6-luna.json').read_text())
            config.update(base_url=f'http://127.0.0.1:{server.server_port}/generation',
                          minimum_request_interval_seconds=0, rate_limit_retry=dict(enabled=False))
            config_path = folder/'provider.json'
            config_path.write_text(json.dumps(config))
            env = dict(os.environ, SCI_JUDGE_API_KEY='local-mock-only',
                       SCI_JUDGE_API_BASE=f'http://127.0.0.1:{server.server_port}/judge')
            env[config['api_key_env']] = 'local-mock-only'

            def cli(*args, expected=0):
                result = subprocess.run([sys.executable, '-m', 'sci', *map(str, args)],
                                        env=env, cwd=folder, capture_output=True, text=True, timeout=600)
                assert result.returncode == expected, (args, result.stdout, result.stderr)
                return result

            out = folder/'batch'
            cli('batch', config_path, questions, out, expected=1)
            predictions = [json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
            assert [p['status'] for p in predictions] == ['answered', 'answered', 'model_error'], predictions
            assert predictions[2]['prediction'] is None
            assert counts == {1: 2, 2: 2, 3: 1}, counts
            schemas = json.loads((ROOT/'configs/tools.json').read_text())
            for request in generation:
                system = next(item for item in request['input'] if item.get('role') in ('system', 'developer'))
                assert system['content'] == (ROOT/'prompts/qa.txt').read_text()
                assert [{k: tool[k] for k in ('name', 'description', 'parameters')} for tool in request['tools']] == schemas
                assert 'FIXTURE_GOLD_' not in json.dumps(request), 'Gold answer leaked into inference'
            assert sum(any(item.get('type') == 'function_call_output' for item in r['input']) for r in generation) == 2
            for row in rows:
                case = out/row['id']
                assert (case/'question.txt').read_text() == row['question']
                assert json.loads((case/'STORAGE_VERIFICATION.json').read_text())['status'] == 'PASS'
            cli('evaluate', questions, out/'predictions.jsonl', folder/'grades', '--model', 'mock-judge')
            grades = [json.loads(line) for line in (folder/'grades/grades.jsonl').read_text().splitlines()]
            assert [g['is_correct'] for g in grades] == [True, False, False]
            assert [g['judge_called'] for g in grades] == [True, True, False]
            assert len(judging) == 2
            assert 'FIXTURE_GOLD_1' in judging[0]['input'][-1]['content']
            assert 'FIXTURE_GOLD_2' in judging[1]['input'][-1]['content']
            summary = json.loads((folder/'grades/summary.json').read_text())
            assert summary['questions'] == 3 and summary['correct'] == 1
            trajectory = json.loads(cli('inspect', out/'fixture-1', '--trajectory').stdout)
            assert any(message['role'] == 'toolResult' for message in trajectory)
            assert len(json.loads(cli('inspect', out/'fixture-1', '--tools').stdout)) == 1
            cli('batch', config_path, questions, out, expected=1)
            assert counts == {1: 2, 2: 2, 3: 1}
            print(json.dumps(dict(status='PASS',synthetic_questions=3, completed=2,
                deliberate_api_failures=1, real_cuda_searches=2, gold_blind_generation=True,
                correct_incorrect_incomplete_grading_checked=True, generation_protocol='Responses',
                judge_protocol='Responses', archive_inspection=True, overwrite_rejected=True,
                paid_api_calls=0, benchmark_accuracy_measured=False), indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == '__main__':
    main()
