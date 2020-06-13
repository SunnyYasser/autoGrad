import unittest
from autograd import tensor

class TestTensor(unittest.TestCase):

    def test_simple_sum(self):
        t1 = tensor.Tensor([1.0 ,2.4, 3], requires_grad=True)
        t2 = t1.sum()
        
        t2.backward()

        assert t1.grad is not None 
    





        
