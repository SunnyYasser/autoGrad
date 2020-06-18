from autograd.tensor import Tensor
import numpy as np

class Parameter(Tensor):
    def __init__(self, *shape) -> None:
        data = np.random.randn(*shape)
        super().__init__(data, requires_grad=True)

    
    

    
