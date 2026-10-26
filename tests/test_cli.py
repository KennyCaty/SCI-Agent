import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from sci.paths import ROOT


class CommandLayout(unittest.TestCase):
    def test_help_routes_from_an_unrelated_working_directory(self):
        commands = [[], ['download'], ['run'], ['batch'], ['evaluate'], ['trec-covid'],
                    ['index'], ['index', 'prepare'], ['index', 'ner'],
                    ['index', 'build'], ['inspect'], ['preflight']]
        with tempfile.TemporaryDirectory() as folder:
            for command in commands:
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable, '-m', 'sci', *command, '--help'],
                                            cwd=folder, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(result.stdout.strip())
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_benchmark_spec_paths_stay_inside_checkout(self):
        spec_path = ROOT / 'configs/benchmarks/trec_covid.json'
        spec = json.loads(spec_path.read_text())
        provider = (spec_path.parent / spec['provider_config']).resolve()
        self.assertEqual(provider, ROOT / 'configs/models/gpt-6-luna.json')
        self.assertTrue(provider.is_file())
        for key, expected in [('assets_root', 'indexes/trec-covid'),
                              ('prepared_data', 'data/trec-covid'),
                              ('embedding_model', 'models/Qwen3-Embedding-0.6B')]:
            self.assertEqual((spec_path.parent / spec[key]).resolve(), ROOT / expected)


if __name__ == '__main__':
    unittest.main()
