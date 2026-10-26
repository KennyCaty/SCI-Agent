import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from sci.benchmark.trec_covid import graded_metrics, parse_ranking, prepare, evaluate, read_qrels, save


class TrecCovid(unittest.TestCase):
    def test_linear_gains_and_unjudged(self):
        q = dict(a=2, b=1, c=0, d=-1)
        self.assertEqual(graded_metrics(['a', 'b'], q)['ndcg_at_10'], 1)
        self.assertAlmostEqual(graded_metrics(['b', 'a'], q)['ndcg_at_10'],
                               (1 + 2/math.log2(3))/(2 + 1/math.log2(3)))
        self.assertEqual(graded_metrics(['a', 'a', 'b'], q), graded_metrics(['a', 'b'], q))
        result = graded_metrics(['d', 'unknown', 'c'], q)
        self.assertEqual(result['ndcg_at_10'], 0)
        self.assertEqual(result['unjudged_returned'], 2)
        self.assertEqual(result['judged_at_10'], .1)

    def test_ranking_ids_headings_and_duplicates(self):
        mapping = {'0': 'abcd', '1': 'efgh'}
        value = parse_ranking('## **Relevant Documents (ranked):**\n1. `.cep/s/0.txt`\n2) source_id=1 — reason\n3. abcd\n4. unknown', mapping)
        self.assertEqual(value, dict(ranked=['abcd', 'efgh'], issues=['duplicate document: abcd', 'unknown document: unknown']))
        self.assertEqual(parse_ranking('no ranking', mapping)['ranked'], [])

    def test_preparation_keeps_qrels_out_of_generation_and_failures_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, assets, out = root/'data', root/'assets', root/'run'
            (data/'qrels').mkdir(parents=True)
            assets.mkdir()
            documents = [dict(_id='abcd', title='Title', text='Body'), dict(_id='efgh', title='Other', text='More')]
            queries = [dict(_id='1', text='test query'), dict(_id='2', text='second query')]
            for name, values in [('corpus', documents), ('queries', queries), ('sources', [dict(source_id='1', relpath='abcd.txt'), dict(source_id='2', relpath='efgh.txt')])]:
                (data/(name+'.jsonl')).write_text('\n'.join(map(json.dumps, values)))
            (data/'qrels/test.tsv').write_text('query-id\tcorpus-id\tscore\n1\tabcd\t2\n1\tefgh\t-1\n2\tefgh\t1\n')
            save(data/'source_id_to_document_id.json', {'1': 'abcd', '2': 'efgh'})
            save(assets/'READY.json', dict(status='PASS'))
            save(root/'provider.json', dict(model='test', api='openai-responses'))
            save(root/'spec.json', dict(provider_config='provider.json', assets_root=str(assets), prepared_data=str(data), embedding_model='/unused/model', fixture=True))
            self.assertEqual(prepare(root/'spec.json', data, out)['questions'], 2)
            self.assertNotIn('abcd', (out/'inputs/1.txt').read_text())
            self.assertNotIn('qrels', (out/'inputs/1.txt').read_text())
            config = json.loads((out/'RUN_CONFIG.json').read_text())
            self.assertEqual(config['embedding']['query_max_length'], 512)
            self.assertEqual(config['task']['type'], 'trec-covid')
            for qid in ('1', '2'):
                (out/'cases'/qid).mkdir()
            save(out/'cases/1/COMPLETE.json', dict(status='answered'))
            (out/'cases/1/final.txt').write_text('Relevant Documents:\n1. source_id=1')
            save(out/'cases/2/COMPLETE.json', dict(status='model_error'))
            result = evaluate(out)
            self.assertEqual((result['answered'], result['failed'], result['ndcg_at_10']), (1, 1, 1))
            self.assertIsNone(result['results'][1]['ndcg_at_10'])
            (out/'cases/1/final.txt').write_text('bad output')
            self.assertEqual(evaluate(out)['ndcg_at_10'], 0)
            (data/'qrels/test.tsv').write_text('query-id corpus-id score\n1 nonexistent 2\n')
            with self.assertRaisesRegex(ValueError, 'Invalid'):
                read_qrels(data/'qrels/test.tsv', {'1'}, {'abcd'})


if __name__ == '__main__':
    unittest.main()
