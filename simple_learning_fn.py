import numpy as np
from autograd.tensor import Tensor

x_data = Tensor(np.random.randn(100,3))
coeff = Tensor(np.array([-1., 3., -2.]))

y_data = x_data@coeff + 5 
print(y_data)

w = Tensor(np.random.randn(3), requires_grad=True)
b = Tensor(np.random.randn(), requires_grad=True)

lr = 0.001
batch_size = 32

for epoch in range(100):
    
    epoch_loss = 0.0

    for start in range(0, 100, batch_size):
        # print(start)
        
        end = start + batch_size
        inputs = x_data[start:end]
        
        w.zero_grad()
        b.zero_grad()

        preds = inputs @ w + b
        actuals = y_data[start:end]
        
        err = preds - actuals
        loss = (err*err).sum()

        loss.backward()

        w -= w.grad * lr
        b -= b.grad * lr

    print(epoch, loss)

