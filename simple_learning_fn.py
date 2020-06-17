import numpy as np
from autograd.tensor import Tensor

x_data = Tensor(np.random.randn(100,3))
coeff = Tensor(np.array([-1., 3., -2.]))

y_data = Tensor(x_data@coeff + 5 + np.random.randint(-2,2, size=(100,)))
print(y_data)

w = Tensor(np.random.randn(3), requires_grad=True)
b = Tensor(np.random.randn(), requires_grad=True)

for epoch in range(100):
    w.zero_grad()
    b.zero_grad()

    # TODO: implment slicing
    # TODO: matrix multiplication
    
    preds = x_data @ w + b
    err = preds - y_data
    loss = (err*err).sum()

    loss.backward()

    w -= w.grad * 0.1
    b -= b.grad * 0.1

    print(epoch, loss)

