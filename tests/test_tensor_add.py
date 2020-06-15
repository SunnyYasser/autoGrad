import unittest
from autograd import tensor

class TestTensorAdd(unittest.TestCase):
    
    def test_simple_add(self):
        t1 = tensor.Tensor([1, 2, 3], requires_grad=True)
        t2 = tensor.Tensor([4, 5 ,6], requires_grad=True)

        # t3 = autograd.tensor.Tensor([])
        # t4 = tensor.add(t1,t2)

        t4 = t1 + t2

        t4.backward(tensor.Tensor([-1, -2, -3]))

        assert t1.grad.data.tolist() == [-1, -2, -3]
        assert t2.grad.data.tolist() == [-1, -2, -3]

    
    def test_broadcast_add1(self):
        t1 = tensor.Tensor([[1., 2 ,3], [4, 5, 6]], requires_grad=True) #(2,3)
        t2 = tensor.Tensor([7, 8, 9], requires_grad=True) #(3,)

        # t3 = tensor.add(t1, t2)
        t3 = t1 + t2 
        t3.backward(tensor.Tensor([[1, 1, 1],[1, 1, 1]]))
        
        assert t3.data.shape == (2,3)
        assert t3.data.tolist() == [[8, 10, 12],[11, 13, 15]]
        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [2, 2, 2]

        
    def test_broadcast_add2(self):
        t1 = tensor.Tensor([[1., 2 ,3], [4, 5, 6]], requires_grad=True) #(2,3)
        t2 = tensor.Tensor([[7, 8, 9]], requires_grad=True) #(1,3)

        # t3 = tensor.add(t1, t2)
        t3 = t1 + t2
        t3.backward(tensor.Tensor([[1, 1, 1],[1, 1, 1]]))
        
        assert t3.data.shape == (2,3)
        assert t3.data.tolist() == [[8, 10, 12],[11, 13, 15]]
        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [[2, 2, 2]]



