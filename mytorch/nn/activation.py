import numpy as np
from mytorch import Tensor

def relu(x: Tensor):
    return Tensor(np.maximum(0, x.numpy()))

