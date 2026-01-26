import numpy as np 
from mytorch import Tensor

def sigmoid(x: Tensor):

    z = np.clip(x._data, -500, 500)
    result = np.zeros_like(z)

    pos_mark = z >=0

    result[pos_mark] = 1 / (1 + np.exp(-z[pos_mark]))


    #bug: when every thing is positive, this get exploded: need to fix this: bug fixed was using pos_mark instead of neg_mark

    neg_mark = z < 0
    result[neg_mark] = np.exp(z[neg_mark]) / (1 + np.exp(z[neg_mark]))  

    return Tensor(result)



def relu(x: Tensor):  ## implementing relu activation function: 
    return Tensor(np.maximum(0, x._data))   # as Relu just changes all the negatives into zeros and and all pos as it is


def softmax(x: Tensor, dim: int = -1) -> Tensor:  # adding a softmax funciton
            ### BEGIN SOLUTION
        # Numerical stability: subtract max to prevent overflow
        x_max_data = np.max(x._data, axis=dim, keepdims=True)
        x_max = Tensor(x_max_data)
        x_shifted = x - x_max  # Tensor subtraction

        # Compute exponentials
        exp_values = Tensor(np.exp(x_shifted._data))

        # Sum along dimension
        exp_sum_data = np.sum(exp_values._data, axis=dim, keepdims=True)
        exp_sum = Tensor(exp_sum_data)

        # Normalize to get probabilities
        result = exp_values / exp_sum
        return result
        ### END SOLUTION



def tanh(x: Tensor): #implemnatiation of tanh
     return Tensor(np.tanh(x._data))


## Gelu implementation: smooth approximation of relu : f(x) = x * Sigmoud(1.702* x)

def gelu(x:Tensor):
     
     sigmoid_part = 1.0 / (1.0 + np.exp(-1.702 * x._data))
     result = x._data * sigmoid_part
     return Tensor(result)



#implemeting the linear layer -> pure math

def linear(x: Tensor, weight: Tensor, bias:Tensor = None) -> Tensor:  
    result = x.matmul(weight)  

    if bias is not None:
        result = result + bias
    return result

