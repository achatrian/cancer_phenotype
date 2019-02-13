"""
Python implementation of the shape matching algorithm "Minimal variance matching"
Latecki, Longin Jan, et al. "An elastic partial shape matching technique." Pattern Recognition 40.11 (2007): 3069-3080.
Author: Andrea Chatrian
"""

__author__ = 'Andrea Chatrian'

import numpy as np


def minimum_variance_matching(a, b):
    m, n = a.shape[0], b.shape[0]
    assert m < n
# FIXME to finish ?
