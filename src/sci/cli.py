"""SCI reproduction commands."""
import argparse
import runpy
import shutil
import subprocess
import sys
from sci.paths import ROOT

COMMANDS = {
    'download': ('sci.indexing.download', 'Download datasets and model files'),
    'run': ('sci.runtime.runner', 'Run one question in a new output directory'),
    'batch': ('sci.benchmark.batch', 'Run QA questions from a JSONL file'),
    'evaluate': ('sci.benchmark.evaluate', 'Judge QA answers separately'),
    'trec-covid': ('sci.benchmark.trec_covid', 'Prepare, run or evaluate TREC-COVID'),
}
INDEX_COMMANDS = {
    'prepare': 'prepare', 'ner': 'ner', 'inputs': 'inputs',
    'build': 'build', 'status': 'status',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Use sci COMMAND --help for command arguments.')
    parser.add_argument('command', choices=[*COMMANDS, 'index', 'inspect', 'preflight'],
                        help='index: construct corpus assets; inspect: read run archives; preflight: check the four tools')
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return
    args = parser.parse_args(argv[:1])
    rest = argv[1:]
    prog = 'sci ' + args.command
    if args.command == 'index':
        index_parser = argparse.ArgumentParser(prog='sci index', description='Index construction, in order: prepare, ner, inputs, build; status reports progress.')
        index_parser.add_argument('stage', choices=INDEX_COMMANDS)
        if not rest or rest[0] in ('-h', '--help'):
            index_parser.print_help()
            return
        stage = index_parser.parse_args(rest[:1]).stage
        prog += ' ' + stage
        rest = rest[1:]
        if stage != 'prepare' and rest:
            stage_parser = argparse.ArgumentParser(prog='sci index '+stage, description='Settings come from SCI_INDEX_CONFIG; default: configs/indexing/browsecomp_plus.json.')
            stage_parser.parse_args(rest)
        module = 'sci.indexing.' + INDEX_COMMANDS[stage]
    elif args.command in ('inspect', 'preflight'):
        if rest and rest[0] in ('-h', '--help'):
            print('sci inspect CASE [--trajectory | --tools | --tool ID | --contexts | --context N | --api N | --artifact PATH]' if args.command == 'inspect' else 'sci preflight (requires a ready local index; no generation API)')
            return
        if args.command == 'preflight' and rest:
            parser.error('preflight takes no arguments; select assets with CEP_RESOURCES_CONFIG')
        node = shutil.which('node')
        if not node:
            parser.error('Node.js is required by the Pi runtime')
        script = 'extensions/inspect.mjs' if args.command == 'inspect' else 'tests/retrieval_preflight.mjs'
        raise SystemExit(subprocess.call([node, str(ROOT/script), *rest]))
    else:
        module = COMMANDS[args.command][0]
    sys.argv = [prog, *rest]
    runpy.run_module(module, run_name='__main__')
