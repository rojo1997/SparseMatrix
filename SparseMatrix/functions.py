from itertools import product
import numpy as np

def cartesian_product(X: list):
    tuples = product(*[
        range(x.start,x.stop,x.step) if isinstance(x,slice) else 
        x if isinstance(x,range) else
        x if isinstance(x,list) else
        [x] for x in X
    ])
    return np.array([t for t in tuples])

def groupby(X: np.array, axis: int or list, method):
    mask = None
    if isinstance(axis,int):
        mask = [i for i in range(X.shape[1]) if i != axis]
    if isinstance(axis,list):
        mask = [i for i in range(X.shape[1]) if i not in axis]
    Z = X[:,axis]
    Y = np.unique(X[:,mask], axis = 0)
    V = np.zeros(shape = (Y.shape[0], len(axis) if isinstance(axis,list) else 1))
    ACC = [np.where(np.isin(X[:,mask],y).prod(axis = 1))[0] for j,y in enumerate(Y)]
    
    for j,index in enumerate(ACC):
        V[j] = method(Z[index], axis = 0)
    return np.hstack((Y,V)), [Y.shape[1] + i for i in range(V.shape[1])]