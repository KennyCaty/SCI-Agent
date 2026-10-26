from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
from sci.indexing.segment import split_text,tokenizer
from sci.indexing.entities import choose_segment


class Segments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.tokenizer=tokenizer()

    def test_exact_token_limits_raw_unicode_and_full_coverage(self):
        samples=['','  \n\t','  Leading\n'+('a short paragraph.\n'*1000)+' trailing  ',
                 '中文字符🙂👩‍🔬和emoji边界。'*2000,'x'*20000,'   '+('longwordwithoutspaces'*1000)+'  ']
        for text in samples:
            for size,overlap in [(512,64),(64,8)]:
                spans=split_text(text,self.tokenizer,size,overlap)
                if not text:self.assertEqual(spans,[]);continue
                self.assertEqual(spans[0][0],0);self.assertEqual(spans[-1][1],len(text))
                for a,b,n in spans:
                    self.assertEqual(n,len(self.tokenizer(text[a:b],add_special_tokens=True)['input_ids']))
                    self.assertLessEqual(n,size)
                self.assertTrue(all(a[0]<b[0] and a[1]>=b[0] for a,b in zip(spans,spans[1:])))

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):split_text('text',self.tokenizer,8,8)

    def test_entity_uses_new_character_spans(self):
        spans=[(0,0,100,'first'),(1,80,180,'second'),(2,160,260,'third')]
        self.assertEqual(choose_segment(spans,95,125),(1,'second'))
        self.assertEqual(choose_segment(spans,0,10),(0,'first'))
        with self.assertRaises(ValueError):choose_segment(spans,300,310)


if __name__=='__main__':unittest.main()

class StructuredSegments(unittest.TestCase):
    def test_headers_count_inside_1024_and_encoding_format_exact(self):
        from sci.indexing.segment_format import format_segment
        from bisect import bisect_right
        tok=tokenizer();title='A long Unicode title 中文🙂 '*8
        text=('  Evidence paragraph with details and punctuation.\r\n'*600)+'尾部内容🙂'
        heading_at=lambda start:'Section A' if start<len(text)//2 else 'Section B'
        spans=split_text(text,tok,1024,128,title=title,heading_at=heading_at)
        self.assertEqual(spans[0][0],0);self.assertEqual(spans[-1][1],len(text))
        for a,b,n in spans:
            actual=format_segment(title,heading_at(a),text[a:b])
            self.assertEqual(n,len(tok(actual,add_special_tokens=True,truncation=False)['input_ids']))
            self.assertLessEqual(n,1024);self.assertTrue(actual.startswith('Title: '));self.assertIn('\nSection: ',actual)
        self.assertTrue(all(a[1]>=b[0] for a,b in zip(spans,spans[1:])))
        self.assertEqual(format_segment('Same','same','  abc  '),'Title: Same\nPassage: abc')
    def test_header_overhead_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):split_text('body '*100,tokenizer(),64,8,title='enormous title '*100)
