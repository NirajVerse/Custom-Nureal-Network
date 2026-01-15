import numpy as np


#creating a Tensor: wrapper around numpy
class Tensor: ## initializing tensor ==> ultimate placeholder for ML neural net and building block 
    

    def __init__(self, data):  #constructor for data and type
        self._data = np.array(data, dtype=np.float32)
        self.dtype = self._data.dtype
        
    @property
    def shape(self):  ## attribute for easy accesss whenever the shape changes
        return self._data.shape
    @property
    def size(self): ## same goes here
        return self._data.size


    def __repr__(self): ## for debugging stuff
        return f'Tensor(data={self._data}, shape={self.shape}, size={self.size})'
    
    def __str__(self): ## for users human readable
        return f'Tensor {self._data}'
    
    #converting back to numpy array
    def numpy(self):
        return self._data
    

    # how memroy in usage: for efficient memory use
    def memory_footprint(self):
        return self._data.nbytes
        
    # adding of tensors with numpy broadcasting
    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._data + other._data)
        else:
            return Tensor(self._data + other)


    # substraction of tensor using numpy broadcasting
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._data - other._data)
        else:
            return Tensor(self._data - other)
        
    # multiplication element-wise - numpy broadcasting
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._data * other._data)
        else:
            return Tensor(self._data * other)
    
    # division 
    def __truediv__(self, other):

        if isinstance(other, Tensor):
            return Tensor(self._data / other._data)
        else:
            return Tensor(self._data / other)
        

    # matrix multiplication of two tensor

    def matmul(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f'Expected Tensor for Matrix Multiplication, got {type(other)}')
        
        # scaler Tensor 0D
        if self.shape == () or other.shape == ():
            return Tensor(self._data * other._data)
        

        # if len(self.shape) == 1 and len(other.shape) == 1: ## vector 1D need to rebuilt it 
        #     return Tensor(np.dot(self._data, other._data))
        
        if len(self.shape) >=2 and len(other.shape) >=2:  ## checing for 2d 
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Cannot Perform matrix multiplication: {self.shape} @ {other.shape}. "
                    f"Inner Dimensions must match: {self.shape[-1]} is not equal to (!=) {other.shape[-2]}"
                )
            

        a = self._data
        b = other._data

        if len(self.shape) == 2 and len(other.shape) ==2: #checking specificially for tensor with size 2 
            M = self.shape[0]
            M2 = other.shape[-1]

            result_data = np.zeros((M,M2), dtype=a.dtype) # creating the matrix of zeros of the required size to store the new one

            for row in range(M):
                for col in range(M2):                                                                          
                    result_data[row,col] = np.dot(a[row, :], b[:, col])  # taking every col and row --> a, b matrix

        else: # tensors with other sizes uses the numpy for matmul

            result_data = np.matmul(a,b) # other tensors
        
        return Tensor(result_data)  # wrapping back in Tensor Class -> Container

            

    def __matmul__(self, other):
        # enabling @ operator for matrix multiplication - wrapper around matmul fun
        return self.matmul(other)



    def __getitem__(self, key):  # slicing and indexing for Tensor - numpy based indexing
        result = self._data[key]
        if not isinstance(result, np.ndarray):  #converting back to numpy array if not
            result = np.array(result)
        return Tensor(result)


    def __setitem__(self, idx, value):  # changing the elements inside the Tensor -> need to fix more, working for 1D
        if isinstance(value, Tensor):
            self._data[idx] = value._data
        else:
            self._data[idx] = value
        return self
    

    def reshape(self, *shape): # changing the dimension of the tensor


        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):   #checking for multiple input (()), ([])
            new_shape = tuple(shape[0])
        else:
            new_shape = shape

        if -1 in new_shape:    # checking for -1 to find the unkown dimension

            if new_shape.count(-1) > 1: ## check if there are more than one -1 in the shape 
                raise ValueError("Can only specify one unknow dimension with -1")
            
            known_size = 1  # calculating the product of the dimensions
            unknown_idx = new_shape.index(-1)  # getting the index of unknown dimension

            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim


            unknown_dim = self.size // known_size  # getting the unknown_dim val
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim # replaing in the newshape
            new_shape = tuple(new_shape)


        if np.prod(new_shape) != self.size:  # raising the error if shape doesnot match
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f'Total Elements must match: {self.size} != {target_size}'
            )

        reshaped_data = np.reshape(self._data, new_shape)  # reshaping the Tensor with numpy and newshape 
        return Tensor(reshaped_data)
    




    def transpose(self, dim0=None, dim1=None):  # col to row  vice-versa
        if dim0 is None and dim1 is None:  # if dimension is not specified 
            if len(self.shape) < 2:
                return Tensor(self._data.copy()) # 0d and 1D tensors is just the copy
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]   # taking the last two dimensions and swapping them 
                transposed_data = np.transpose(self._data, axes)
        else:
            if dim0 is None or dim1 is None:
                raise ValueError(f'Both dim0 and dim1 must be specified')  ## if any dimension is not specified then 
            axes = list(range(len(self._data)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self._data, axes)

        return Tensor(transposed_data)
    


    def sum(self, axis=None, keepdim=False): ## sum of the matrix with dim and keepdim
        if axis is None:
            return Tensor(np.sum(self._data))
        
        if isinstance(axis, int):
            dim = (axis,)
            return Tensor(np.sum(self._data, axis=dim, keepdims=keepdim)) # using numpy.sum()


    def mean(self, axis=None, keepdim=False):  #  repeating same for mean
        result = np.mean(self._dataa, axis=axis, keepdims=keepdim)
        return Tensor(result)
    
    def max(self, axis=None, keepdim=False):  # repeating same for max
        result = np.max(self._data, axis=axis, keepdims=keepdim)
        return Tensor(result)