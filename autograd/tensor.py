import numpy as np 
from typing import List, NamedTuple, Callable

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]


class Tensor:
    """
        keeps a track of the previous gradients
    """

    def __init__(self,
                data: np.ndarray,
                requires_grad:bool = False,
                depends_on: List[Dependency] = None) -> None:

                self.data = data
                self.requires_grad = requires_grad
                self.depends_on = depends_on
                self.shape = data.shape

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    def sum(self) -> 'Tensor':
        return tensor_sum(self)

    def backward(self, grad:'Tensor') -> 'Tensor':
        if self.requires_grad:
            raise NotImplementedError
        else
            return None


def tensor_sum(t: Tensor) -> Tensor:
    """
    Takes a tensor and returns the 0-tensor,i.e, the sum
    """
    data = t.data.sum()
    requires_grad = t.requires_grad


    if requires_grad:
        
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            Grad is necessarily a zero tensor
            """

            return grad*np.ones_like(t.data)


        depends_on = [Dependency(t, grad_fn)]
    
    else:
        depends_on = None

    return Tensor(data,requires_grad=requires_grad,depends_on=depends_on)


