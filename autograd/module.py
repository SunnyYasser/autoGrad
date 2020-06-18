import inspect
import numpy as np
from typing import Iterator

from autograd.tensor import Tensor
from autograd.parameters import Parameter

class Module:
    
    def parameters(self) -> Iterator[Parameter]:
        for _, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
    
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()
    
    



