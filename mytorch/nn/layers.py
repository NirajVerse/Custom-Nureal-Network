import numpy as np
from mytorch import Tensor
from . module import Module
from . import functional as F

# Constants for weight initialization
XAVIER_SCALE_FACTOR = 1.0  # Xavier/Glorot initialization uses sqrt(1/fan_in)
HE_SCALE_FACTOR = 2.0  # He initialization uses sqrt(2/fan_in) for ReLU

# Constants for dropout
DROPOUT_MIN_PROB = 0.0  # Minimum dropout probability (no dropout)
DROPOUT_MAX_PROB = 1.0  # Maximum dropout probability (drop everything)


class Linear(Module):

    def __init__(self, input_feature:int, output_feature:int, bias: bool =True):

        #initialization of random weights and bias for the start

        self.input_feature = input_feature
        self.output_feature = output_feature

        #xavier initialization for stable gradient
        scale = np.sqrt(XAVIER_SCALE_FACTOR / input_feature)
        scale_weight = np.random.randn(output_feature, input_feature) * scale
        self.weight = Tensor(scale_weight, requires_grad = True)

        #initialization for bias
        if bias:
            bias_data = np.zeros(output_feature)
            self.bias = Tensor(bias_data, requires_grad = True)

        else:
            self.bias = None


    def forward(self, x:Tensor) ->Tensor:
        return F.linear(x, self.weight, self.bias)  #using functional api for computation
    
    def parameters(self): ## returning learnable parameters

        if self.bias is not None:
            return [self.weight, self.bias]
        else:
            return [self.weight]


    def __repr__(self):
        """String representation for debugging."""
        bias_str = f", bias={self.bias is not None}"
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}{bias_str})" 
    
