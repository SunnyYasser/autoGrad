from autograd.tensor import Tensor, mul

x = Tensor([10, -10, 10, -5, 6, 3, 1], requires_grad=True)

for i in range(100):
    sum_of_squares = mul(x, x).sum() #sum_of_squares is a zero tensor
    sum_of_squares.backward()

    delta_x = Tensor(0.1 * x.grad.data, requires_grad=x.requires_grad)
    x = Tensor(x.data - delta_x.data, requires_grad=True)

    print(i, x)


