import unittest
from autograd import tensor

class TestTensorMul(unittest.TestCase):
    
    def test_simple_mul(self):
        t1 = tensor.Tensor([1, 2, 3], requires_grad=True)
        t2 = tensor.Tensor([4, 5 ,6], requires_grad=True)

        # t3 = autograd.tensor.Tensor([])
        t4 = t1 * t2

        t4.backward(tensor.Tensor([-1, -2, -3]))

        assert t4.data.tolist() == [4, 10, 18]
        assert t1.grad.data.tolist() == [-4, -10, -18]
        assert t2.grad.data.tolist() == [-1, -4, -9]

    
    def test_broadcast_mul1(self):
        t1 = tensor.Tensor([[1., 2 ,3], [4, 5, 6]], requires_grad=True) #(2,3)
        t2 = tensor.Tensor([7, 8, 9], requires_grad=True) #(3,)

        # t3 = tensor.mul(t1, t2)
        t3 = t1 * t2
        t3.backward(tensor.Tensor([[1, 1, 1],[1, 1, 1]]))
        
        assert t3.data.shape == (2,3)
        assert t3.data.tolist() == [[7, 16, 27],[28, 40, 54]]
        assert t1.grad.data.tolist() == [[7, 8, 9], [7, 8, 9]]
        assert t2.grad.data.tolist() == [5, 7, 9]


        
    def test_broadcast_mul2(self):
        t1 = tensor.Tensor([[1., 2 ,3], [4, 5, 6]], requires_grad=True) #(2,3)
        t2 = tensor.Tensor([[7, 8, 9]], requires_grad=True) #(1,3)

        # t3 = tensor.mul(t1, t2)
        t3 = t1 * t2 
        t3.backward(tensor.Tensor([[1, 1, 1],[1, 1, 1]]))
        
        assert t3.data.shape == (2,3)
        assert t3.data.tolist() == [[7, 16, 27],[28, 40, 54]]
        assert t1.grad.data.tolist() == [[7, 8, 9], [7, 8, 9]]
        assert t2.grad.data.tolist() == [[5, 7, 9]]



