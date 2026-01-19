from mytorch import Tensor
from mytorch.nn import relu

x = Tensor([-1,-2,3,4])
y = Tensor([-5,-6,-4,7])
result = x + y
z = relu(result)
print(z)