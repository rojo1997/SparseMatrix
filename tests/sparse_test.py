import unittest
import numpy as np

import sys
sys.path[0] = sys.path[0].replace('/tests','')

from SparseMatrix.SparseMatrix import SparseMatrix

class SparseMatrixTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SparseMatrixTest, self).__init__(*args, **kwargs)

    def test_from_numpy(self):
        size = (30,30,30)
        A = np.random.randint(1,10, size = size)
        M = SparseMatrix.from_numpy(A)
        assert M.T.shape[0] == np.array(size).prod(), 'No tienen el mismo numero de elementos.'
        assert M.sum() == A.sum(), 'Error en la suma de elementos.'
        pass

    def test_randint(self):
        low = 0
        high = 10
        sparsity = 0.01
        shape = (100,100,100)
        fill_value = 0.0
        M = SparseMatrix.randint(low = low, high = high, sparsity = sparsity, shape = shape, fill_value = fill_value)
        assert M.min() >= low, 'No cumple el mínimo'
        assert M.max() < high, 'No cumple el mínimo'
        n = float(np.array(shape).prod())
        assert (M.T.shape[0]/n) < 0.02 and (M.T.shape[0]/n) > 0.00, 'Dispersión cumplida'

if __name__ == "__main__":
    unittest.main(verbosity = 2)