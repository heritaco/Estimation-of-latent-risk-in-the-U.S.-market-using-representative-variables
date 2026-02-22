import numpy as np
from guerrero_trend import difference_matrix

def test_difference_matrix_d1():
    K = difference_matrix(5, 1)
    x = np.array([0, 1, 4, 9, 16], dtype=float)
    y = K @ x
    assert np.allclose(y, np.array([1,3,5,7], dtype=float))

def test_difference_matrix_d2():
    K = difference_matrix(6, 2)
    x = np.array([0, 1, 4, 9, 16, 25], dtype=float)
    y = K @ x
    assert np.allclose(y, np.array([2,2,2,2], dtype=float))
