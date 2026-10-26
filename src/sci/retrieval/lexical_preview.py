"""BM25 preview selection within retrieved segments. Offsets count Unicode characters."""
import bisect
import math
import re
from collections import defaultdict

import Stemmer
from bm25s.stopwords import STOPWORDS_EN


class LexicalPreview:
    def __init__(self, vocabulary, indptr, num_docs):
        self.vocabulary = vocabulary
        self.indptr = indptr
        self.num_docs = num_docs
        self.stemmer = Stemmer.Stemmer('english')
        self.stopwords = set(STOPWORDS_EN)

    def tokens(self, text):
        matches = list(re.finditer(r'\b\w\w+\b', text, re.UNICODE))
        stems = self.stemmer.stemWords([m.group().lower() for m in matches])
        return [(stem, m.start(), m.end()) for stem, m in zip(stems, matches)
                if m.group().lower() not in self.stopwords]

    def idf(self, term):
        index = self.vocabulary.get(term)
        # Use df=1 for terms absent from the corpus vocabulary.
        df = max(1, int(self.indptr[index + 1] - self.indptr[index])) if index is not None else 1
        return math.log1p((self.num_docs - df + .5) / (df + .5))

    @staticmethod
    def aligned_start(body, start, lower, upper, size):
        start = max(lower, min(start, max(lower, upper - size)))
        # Align to a nearby sentence boundary within the segment.
        left = max(lower, start - 60)
        boundaries = [m.end() + left for m in re.finditer(r'(?:[.!?]\s+|\n+)', body[left:start])]
        if boundaries:
            start = boundaries[-1]
        elif start > lower and start < upper and body[start - 1].isalnum() and body[start].isalnum():
            m = re.search(r'\s', body[start:min(upper, start + 32)])
            if m:
                start += m.end()
        return max(lower, min(start, max(lower, upper - size)))

    def select(self, body, query, segments, size=600):
        """Select a raw window from up to two segments ordered by dense score."""
        assert 0 < len(segments) <= 2
        assert size in (240, 600)
        terms = set(t for t, _, _ in self.tokens(query))
        weights = {t: self.idf(t) for t in terms}
        candidates = {}
        for rank, segment in enumerate(segments):
            lower, upper = segment['start'], segment['end']
            assert 0 <= lower < upper <= len(body)
            tokens = [(t, a + lower, b + lower) for t, a, b in self.tokens(body[lower:upper])]
            matches = [(t, a, b) for t, a, b in tokens if t in terms]
            if not matches:
                continue
            positions = defaultdict(list)
            for term, start, end in matches:
                positions[term].append((start, end))
            starts = {lower, max(lower, upper - size)}
            for _, start, _ in matches:
                starts.update(start - shift for shift in (100, 300, 500))
            token_starts = [a for _, a, _ in tokens]
            for start in sorted(starts):
                start = self.aligned_start(body, start, lower, upper, size)
                end = min(upper, start + size)
                key = (start, end)
                if key in candidates:
                    continue
                counts = {}
                for term, hits in positions.items():
                    n = sum(a >= start and b <= end for a, b in hits)
                    if n:
                        counts[term] = n
                if not counts:
                    continue
                length = max(1, bisect.bisect_left(token_starts, end) - bisect.bisect_left(token_starts, start))
                candidates[key] = dict(start=start, end=end, segment=segment,
                                       segment_rank=rank, counts=counts, length=length)
        if not candidates:
            first = segments[0]
            return dict(start=first['start'], end=min(first['end'], first['start'] + size),
                        segment=first, lexical_score=0.0, matched_terms=[],
                        fallback=True, candidate_count=0)
        average = sum(c['length'] for c in candidates.values()) / len(candidates)
        for c in candidates.values():
            norm = 1.5 * (.25 + .75 * c['length'] / average)
            c['lexical_score'] = sum(weights[t] * n * 2.5 / (n + norm) for t, n in c['counts'].items())
        # Break score ties by segment rank, then character offset.
        best = min(candidates.values(), key=lambda c: (-c['lexical_score'], c['segment_rank'], c['start']))
        return dict(start=best['start'], end=best['end'], segment=best['segment'],
                    lexical_score=best['lexical_score'], matched_terms=sorted(best['counts']),
                    fallback=False, candidate_count=len(candidates))
