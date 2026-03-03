"""Tests for factorization utilities."""

import numpy as np
from exotic_compress.compress_tt import factorize_shape


def test_factorize_768():
    factors = factorize_shape(768)
    assert np.prod(factors) == 768
    assert len(factors) <= 4


def test_factorize_3072():
    factors = factorize_shape(3072)
    assert np.prod(factors) == 3072
    assert len(factors) <= 4


def test_factorize_small():
    factors = factorize_shape(12)
    assert np.prod(factors) == 12
