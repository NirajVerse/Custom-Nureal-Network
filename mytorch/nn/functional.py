import numpy as np 
from mytorch import Tensor

def sigmoid(x: Tensor):

    z = np.clip(x._data, -500, 500)
    result = np.zeros_like(z)

    pos_mark = z >=0

    result[pos_mark] = 1 / (1 + np.exp(-z[pos_mark]))

    neg_mark = z < 0
    result[neg_mark] = np.exp(z[pos_mark]) / (1 + np.exp(z[pos_mark]))

    return Tensor(result)