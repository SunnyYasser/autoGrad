import numpy as np 
from typing import List, NamedTuple, Callable, Optional, Union


class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

"""
To make different data type handling easier
"""

Arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable:Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)




class Tensor:

    def __init__(self,
                data: Arrayable,
                requires_grad:bool = False,
                depends_on: List[Dependency] = None) -> None:

        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()
        

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype = np.float64))
    
    def sum(self) -> 'Tensor':
        return tensor_sum(self)

    def backward(self, grad:'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on a non-required_grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        
        # print("grad data", grad.data)
        # print("self.grad.data", self.grad.data)

        self.grad.data += grad.data
        # will have to change later for more flexibility

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

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
        depends_on = []

    return Tensor(data,requires_grad=requires_grad,depends_on=depends_on)


def add(t1: Tensor, t2: Tensor) -> Tensor:
    """
    Takes two tensors and returns their sum
    """
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            
            # Sum added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis = 0)

            # sum across broadcasted dims
            for i,dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        
        depends_on.append(Dependency(t1,grad_fn1))

    if t2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            
            #sum added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis = 0)

            #sum across broadcasted dims            
            for i,dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        
        depends_on.append(Dependency(t2,grad_fn2))
    
    return Tensor(data, requires_grad, depends_on)