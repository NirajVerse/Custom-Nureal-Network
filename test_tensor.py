from mytorch import Tensor
import mytorch.nn as nn

x = Tensor([-1,-2,3,4])
y = Tensor([-5,-6,-4,7])
result = x + y
z = nn.sigmoid(result)
print(z)