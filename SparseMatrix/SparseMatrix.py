import numpy as np
from itertools import product
from functools import reduce

from SparseMatrix.functions import (
    cartesian_product,
    groupby
)

class SparseMatrix:
    def __init__(self, shape: list or tuple, dtype = np.float64, fill_value = 0.0):
        self.shape = np.array(shape)
        self.dtype = dtype
        self.fill_value = fill_value
        self.T = np.zeros(
            shape = (0,self.shape.shape[0] + 1), 
            dtype = self.dtype
        )

    @classmethod
    def from_numpy(cls, array: np.array, fill_value = 0.0):
        M = SparseMatrix(
            shape = array.shape,
            dtype = array.dtype,
            fill_value = fill_value
        )
        M.T = np.vstack((np.where(array != fill_value),array[array > fill_value])).T
        return M

    @classmethod
    def randint(cls, low: int, high: int, sparsity: float, shape: tuple or list, fill_value = 0.0):
        M = SparseMatrix(shape = shape, fill_value = fill_value)
        n = int(np.floor(np.array(shape).prod() * sparsity))
        M.T = np.vstack([np.random.randint(0,shape[i], size = n) for i in range(len(shape))] + [np.random.randint(low,high,n)]).T
        return M

    def __getitem__(self, args):
        index,args = self.__getindex__(args)
        mask = np.array([isinstance(a,slice) for a in args])
        if mask.shape[0] == 0:
            return self.fill_value
        elif index.shape[0] == 1:
            return self.T[index[0],-1]
        else:
            mins = np.array([a.start for a in args[mask]])
            maxs = np.array([a.stop for a in args[mask]])
            steps = np.array([a.step for a in args[mask]])
            M = SparseMatrix(shape = np.ceil((maxs - mins) / steps).astype(np.int64))
            if index.shape[0] != 0:
                M.T = self.T[:,np.where(np.append(mask,True))[0]][index]
                M.T[:,:-1] = np.floor((M.T[:,:-1] - mins) / steps).astype(np.int64)
            return M

    def __getindex__(self, args):
        self.__check__(args)
        args = [slice(
            a.start if a.start != None else 0,
            a.stop if a.stop != None else self.shape[i],
            a.step if a.step != None else 1
        ) if isinstance(a,slice) else a for i,a in enumerate(args)]
        args = np.array(args)
        lindex = [
            np.where(
                (self.T[:,i] >= a.start) &
                (self.T[:,i] < a.stop) &
                (np.mod(self.T[:,i] - a.start, a.step) == 0)
            )[0] if isinstance(a,slice) else 
            np.where(
                self.T[:,i] == a
            )[0] for i,a in enumerate(args)
        ]
        index = reduce(np.intersect1d,lindex)
        return index,args

    def __setitem__(self, args, value):
        index,args = self.__getindex__(args)
        if isinstance(value,np.ndarray):
            args_index = cartesian_product(args)
            X = value.ravel().reshape(len(value),1)
            self.T = np.append(
                self.T,
                np.append(args_index, X, axis = 1),
                axis = 0
            )
        elif isinstance(value, SparseMatrix):
            pass
        else:
            self.T = np.delete(self.T, index, axis = 0)
            if value != self.fill_value:
                if (index.shape[0] == 0) or (index.shape[0] == 1):
                    self.T = np.vstack([self.T,np.append(args,value)])
                else:
                    new_index = cartesian_product(args)
                    new_T = np.append(
                        new_index,
                        np.repeat(value,new_index.shape[0]).reshape(new_index.shape[0],1),
                        axis = 1
                    )
                    self.T = np.append(self.T, new_T, axis = 0)

    def __check__(self, args: np.array):
        assert len(args) == len(self.shape), 'Error número de índices'
        #assert np.all((args < self.shape) if not isinstance(a,slice)])), 'Fuera de dimensión'

    def __str__(self):
        string = '<SparseMatrix: shape={shape}, dtype={dtype}, n0v={n0v}, fill_value={fill_value}>'.format(
            shape = self.shape, 
            dtype = self.dtype, 
            n0v = self.T.shape[0],
            fill_value = self.fill_value
        )
        return string

    def to_numpy(self):
        X = np.zeros(shape = self.shape) + self.fill_value
        for row in self.T:
            X[tuple(row[:-1].astype(int))] = row[-1]
        return X

    def max(self, axis = None):
        if axis == None:
            return np.append(self.T[:,-1],self.fill_value).max()
        else:
            mask = [i for i in range(self.shape.shape[0]) if i != axis]
            M = SparseMatrix(shape = self.shape[mask])
            M.T,_ = groupby(self.T,axis = axis, method = np.max)
            return M
            

    def min(self, axis = None):
        if axis == None:
            return np.append(self.T[:,-1],self.fill_value).min()
        else:
            pass

    def mean(self, axis = None):
        if axis == None:
            n = float(self.shape.prod())
            n0v = float(self.T.shape[0])
            return (n0v/n) * self.T[:,-1].mean() + ((n - n0v)/n) * self.fill_value
        else:
            pass
    
    def std(self, axis = None):
        if axis == None:
            return np.sqrt(self.var())
        else:
            pass

    def var(self, axis = None):
        if axis == None:
            mean = self.mean()
            n = float(self.shape.prod())
            return (np.power(self.T[:,-1] - mean,2).sum() + (self.fill_value - mean)**2 * (n - self.T.shape[0])) / n
        else:
            pass

    def sum(self, axis = None):
        return self.T[:,-1].sum() + (self.shape.prod() - self.T.shape[0]) * self.fill_value

    def __add__(self,M):
        if isinstance(M,np.ndarray):
            pass
        elif isinstance(M,SparseMatrix):
            pass
        else:
            out = self.copy()
            out.fill_value += M
            out.T[:,-1] = out.T[:,-1] + M
            return out

    def copy(self):
        M = SparseMatrix(shape = self.shape, fill_value = self.fill_value)
        M.T = self.T.copy()
        return M

    def abs_frecuency(self):
        abs_f = [(self.fill_value,(self.shape.prod() - self.T.shape[0]))]
        for x in np.unique(self.T[:,-1]):
            abs_f.append((x,(self.T[:,-1] == x).sum()))
        return np.array(abs_f)
    
    def dot(self,X):
        if isinstance(X,np.ndarray):
            pass
        elif isinstance(X,SparseMatrix):
            pass
        else:
            pass

    def multiply(self,X):
        pass