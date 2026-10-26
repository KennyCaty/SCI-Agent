import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from sci.retrieval.exact_topk import topk, order


class ExactSelection(unittest.TestCase):
    def test_ties_masks_and_nan_match_full_sort(self):
        rng = np.random.default_rng(2026)
        ids = rng.permutation(5000)
        for values in (rng.integers(0, 5, 5000).astype(float), np.zeros(5000), rng.normal(size=5000)):
            for mask in (None, values > 0):
                for count in (1, 20, 1000, 6000):
                    np.testing.assert_array_equal(topk(values, ids, count, mask), order(values, ids, mask)[:count])
        values[42] = np.nan
        np.testing.assert_array_equal(topk(values, ids, 1000), order(values, ids)[:1000])


class CudaBackground(unittest.TestCase):
    def test_background_matches_scipy_with_dangling_nodes(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        from sci.retrieval.cuda_runtime import DEVICE, csr_cuda
        from sci.retrieval.background import compute_background
        # Includes dangling columns and asymmetric edges.
        matrix = sparse.csr_matrix(np.array([[0, .7, 0, 0], [.2, 0, 0, 0], [.8, .3, 0, 0], [0, 0, 0, 0]], dtype=np.float32))
        g = np.array([.12, .11, .10, .09], dtype=np.float32)
        z = matrix.T @ g
        denominator = np.where(z <= 0, 1., z)
        b0 = np.full(4, .25, dtype=np.float32)
        expected = b0.copy()
        for _ in range(24):
            expected = .5*b0 + .5*(g*(matrix @ (expected/denominator)) + b0*expected[z <= 0].sum())
        r = SimpleNamespace(n=4, matrix=matrix, t=csr_cuda(matrix), tt=csr_cuda(matrix.T))
        actual = compute_background(r, torch.as_tensor(g, device=DEVICE))
        for observed, reference in zip(actual, (expected[:, None], denominator, 1/np.sqrt(expected))):
            np.testing.assert_allclose(observed.cpu().numpy(), reference, rtol=2e-5, atol=1e-7)
        self.assertTrue(all(a.is_cuda for a in actual))


if __name__ == '__main__':
    unittest.main()
