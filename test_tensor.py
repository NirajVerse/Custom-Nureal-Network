from mytorch import Tensor
import mytorch.nn as nn

from mytorch.nn import functional as F
from mytorch.nn import Sigmoid, ReLU, Softmax, Tanh, GELU
from mytorch.nn import Linear


x = Tensor([[1,2,3]])

linear = Linear(2,3)
print(linear)
out = linear(x)
print(out)