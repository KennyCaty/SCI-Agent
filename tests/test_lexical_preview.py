import sys
import unittest
from pathlib import Path

from sci.retrieval.lexical_preview import LexicalPreview


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.selector = LexicalPreview({}, [0], 100195)

    def segment(self, body, start=0, end=None, row=1):
        return dict(start=start, end=len(body) if end is None else end, seq=row, row=row, score=1 / row)

    def test_evidence_beyond_prefix(self):
        body = 'Shopping and subscription information. ' * 25 + '\nThe album lists track six as a dance song.\n' + 'Footer. ' * 70
        result = self.selector.select(body, 'album track dance song', [self.segment(body)])
        self.assertGreater(result['start'], 0)
        self.assertIn('track six', body[result['start']:result['end']])
        self.assertLessEqual(result['end'] - result['start'], 600)

    def test_second_segment_can_supply_window(self):
        body = 'General music information. ' * 30 + '\nA violin quartet received a composer award. ' * 8
        split = body.index('A violin')
        result = self.selector.select(body, 'violin quartet composer award', [self.segment(body, end=split), self.segment(body, start=split, row=2)])
        self.assertEqual(result['segment']['row'], 2)

    def test_no_match_falls_back_to_semantic_first(self):
        body = 'No shared vocabulary here. ' * 60
        result = self.selector.select(body, 'xylophones', [self.segment(body, start=80)])
        self.assertTrue(result['fallback'])
        self.assertEqual((result['start'], result['end']), (80, 680))

    def test_stemming_and_word_boundaries(self):
        body = 'The researcher studies filters and filtered drops. ' * 30
        result = self.selector.select(body, 'filtering', [self.segment(body)])
        self.assertFalse(result['fallback'])
        self.assertEqual(result['matched_terms'], ['filter'])
        result = self.selector.select('The artichoke grew.', 'art', [self.segment('The artichoke grew.')])
        self.assertTrue(result['fallback'])

    def test_unicode_offsets_and_short_source(self):
        body = 'İstanbul 😀 research on filtration.'
        result = self.selector.select(body, 'filtration', [self.segment(body)])
        self.assertIn('filtration', body[result['start']:result['end']])
        self.assertEqual(result['end'], len(body))

    def test_no_cross_segment_boundary_and_deterministic(self):
        body = 'prefix ' * 100 + 'target evidence. ' * 60 + 'suffix ' * 100
        segment = self.segment(body, start=700, end=1660)
        first = self.selector.select(body, 'target evidence', [segment])
        self.assertEqual(first, self.selector.select(body, 'target evidence', [segment]))
        self.assertGreaterEqual(first['start'], 700)
        self.assertLessEqual(first['end'], 1660)


if __name__ == '__main__':
    unittest.main()
