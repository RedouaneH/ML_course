# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    N = len(x)
    X = np.c_[np.ones(N), x]

    if degree > 1:
        for d in range(2, degree+1):
            X = np.c_[X, x**d]

    return X