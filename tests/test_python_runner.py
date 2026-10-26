import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from sci.benchmark.batch import batch
from sci.runtime.runner import run


class PythonRunner(unittest.TestCase):
    def test_batch_questions_only_and_failure_is_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            questions = root/'questions.jsonl'
            questions.write_text('\n'.join(json.dumps(dict(id=str(i), question='question '+str(i), answer='GOLD_SENTINEL')) for i in range(2)))
            seen = []
            def fake_run(config, question, output, **options):
                seen.append(question.read_text())
                output.mkdir()
                if len(seen) == 2:
                    raise RuntimeError('deliberate fixture failure')
                (output/'final.txt').write_text('prediction')
                return dict(status='answered')
            with patch('sci.benchmark.batch.run', side_effect=fake_run):
                results = batch('unused', questions, root/'run')
            self.assertEqual(seen, ['question 0', 'question 1'])
            self.assertEqual([r['status'] for r in results], ['answered', 'error'])
            self.assertIsNone(results[1]['prediction'])
            self.assertEqual(len((root/'run/predictions.jsonl').read_text().splitlines()), 2)
            with self.assertRaises(FileExistsError):
                batch('unused', questions, root/'run')

    def test_unsafe_ids_rejected_before_outputs_created(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            q = root/'q.jsonl'
            q.write_text(json.dumps(dict(id='../escape', question='question')))
            with self.assertRaises(ValueError):
                batch('unused', q, root/'run')
            self.assertFalse((root/'run').exists())

    def test_single_attempt_existing_directory_not_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root/'marker'
            marker.write_text('keep')
            with self.assertRaises(FileExistsError):
                run(root/'unused-config', root/'unused-question', root)
            self.assertEqual(marker.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
