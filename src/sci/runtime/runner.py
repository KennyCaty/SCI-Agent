"""Run one question with the Pi SDK."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from sci.paths import ROOT


def runtime_args(parser):
    parser.add_argument('--request-interval-seconds', type=float)
    parser.add_argument('--api', choices=['openai-responses', 'openai-completions'])


def runtime_options(args):
    options = {}
    interval = args.request_interval_seconds
    if interval is not None:
        if not math.isfinite(interval) or interval < 0:
            raise ValueError('Request interval must be finite and nonnegative')
        options['requestIntervalSeconds'] = interval
    if args.api is not None:
        options['api'] = args.api
    return options


def run(config, question, output, *, resume=None, **options):
    config, question, output = (Path(p).resolve() for p in (config, question, output))
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    json.loads(config.read_text())
    if not question.read_text().strip():
        raise ValueError('Empty question')
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Node.js is required to run Pi')
    if resume:
        options['resumeFrom'] = str(Path(resume).resolve())
    env = dict(os.environ)
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    request = dict(id=1, type='prompt', config=str(config), question=str(question),
                   output=str(output), options=options)
    process = subprocess.Popen([node, str(ROOT / 'extensions/pi_rpc.mjs')], cwd=ROOT,
                               env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               text=True, encoding='utf-8', start_new_session=True)
    handlers = {}
    def stop(signum, _frame):
        if process.poll() is None:
            os.killpg(process.pid, signum)
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, stop)
        stdout, _ = process.communicate(json.dumps(request) + '\n')
        responses = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        if len(responses) != 1 or responses[0].get('id') != 1:
            raise RuntimeError('Pi RPC ended without one matching response; inspect case logs')
        response = responses[0]
        if process.returncode or not response.get('success'):
            raise RuntimeError(response.get('error', 'Pi process failed'))
        return response['result']
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('question')
    parser.add_argument('output')
    parser.add_argument('--resume')
    runtime_args(parser)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.question, args.output,
                         resume=args.resume, **runtime_options(args)), indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
