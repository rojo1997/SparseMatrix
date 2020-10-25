import unittest
import numpy as np

import sys
sys.path[0] = sys.path[0].replace('/tests','')

from SparseMatrix.SparseMatrix import SparseMatrix

from SparseMatrix.functions import groupby

class SparseMatrixTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SparseMatrixTest, self).__init__(*args, **kwargs)
        self.low = 0
        self.high = 10
        self.sparsity = 0.01
        self.shape = (100,100,100)
        self.fill_value = 0.0
        self.A = np.random.randint(
            low = self.low,
            high = self.high, 
            size = self.shape,
        )

    def test_from_numpy(self):
        M = SparseMatrix.from_numpy(self.A)
        assert M.shape.prod() == np.array(self.shape).prod(), 'No tienen el mismo numero de elementos.'
        assert M.sum() == self.A.sum(), 'Error en la suma de elementos.'
        assert M.min() >= self.low, 'No cumple el mínimo'
        assert M.max() < self.high, 'No cumple el máximo'

    def test_to_numpy(self):
        M = SparseMatrix.from_numpy(self.A)
        M = M.to_numpy()
        assert np.all(M == self.A), 'No cumple el máximo'

    def test_randint(self):
        M = SparseMatrix.randint(
            self.low, 
            self.high, 
            self.sparsity, 
            self.shape, 
            self.fill_value
        )
        assert M.min() >= self.low, 'No cumple el mínimo'
        assert M.max() < self.high, 'No cumple el máximo'
        n = float(np.array(self.shape).prod())
        assert (M.T.shape[0]/n) == self.sparsity, 'Dispersión cumplida'

    def test_sum(self):
        M = SparseMatrix.from_numpy(self.A)
        assert M.sum() == self.A.sum(), 'No cumple la suma'

    def test_min(self):
        M = SparseMatrix.from_numpy(self.A)
        assert M.min() == self.A.min(), 'No cumple el mínimo'

    def test_max(self):
        M = SparseMatrix.from_numpy(self.A)
        assert M.max() == self.A.max(), 'No cumple el máximo'

    def test_mean(self):
        M = SparseMatrix.from_numpy(self.A)
        assert round(M.mean(),5) == round(self.A.mean(),5), 'No cumple la media'

    def test_std(self):
        M = SparseMatrix.from_numpy(self.A)
        assert round(M.std(),5) == round(self.A.std(),5), 'No cumple la desviación típica'

    def test_var(self):
        M = SparseMatrix.from_numpy(self.A)
        assert round(M.var(),5) == round(self.A.var(),5), 'No cumple la varianza'
    
    def test_setitem_one_vs_one(self):
        M = SparseMatrix.from_numpy(self.A)
        for i in range(100):
            self.A[i,i,i] = -1000
            M[i,i,i] = -1000
            assert self.A[i,i,i] == M[i,i,i], 'Error en la asignación de 1 elemento'
    
    def test_setitem_all_one(self):
        M = SparseMatrix.from_numpy(self.A)
        M[:,:,:] = -1000
        assert M.mean() == -1000, ''

    def test_add_int(self):
        M = SparseMatrix.from_numpy(self.A)
        assert np.all((M + 10).to_numpy() == (self.A + 10)), 'Error suma entero'

    def test_abs_frecuenty(self):
        M = SparseMatrix(shape = (10,10,10))
        M[1,2,3] = 2
        M[3,2,1] = 2
        M[2,2,2] = 2
        R = M.abs_frecuency()
        assert (R[0,1] == 997) and (R[1,1] == 3), 'Error en la frecuencia'
    
    def test_max_axis(self):
        M = SparseMatrix.from_numpy(self.A)
        print(M.max(axis = 1).shape)

class FunctionTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(FunctionTest, self).__init__(*args, **kwargs)

    def test_groupby(self):
        X = np.random.randint(0,10,size = (10,5))
        Y,index = groupby(X, axis = 3, method = np.sum)
        print(Y)
        print(index)


if __name__ == "__main__":
    unittest.main(verbosity = 2)