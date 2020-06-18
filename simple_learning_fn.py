import numpy as np
from autograd.tensor import Tensor
from autograd.parameters import Parameter
from autograd.module import Module
from autograd.optim import SGD

x_data = Tensor(np.random.randn(100,3))
coeff = Tensor(np.array([-1., 3., -2.]))
y_data = x_data@coeff + 5 

class Model(Module):
    def __init__(self) -> None:
        self.w  = Parameter(3)
        self.b  = Parameter()

    def predict(self, x:Tensor) -> Tensor:
        return x @ self.w + self.b


model = Model()    
lr = 0.001
batch_size = 32
optimizer = SGD(lr)


for epoch in range(100):
    
    epoch_loss = 0.0

    for start in range(0, 100, batch_size):
        # print(start)
        
        end = start + batch_size
        inputs = x_data[start:end]
        model.zero_grad()

        preds = model.predict(inputs)
        actuals = y_data[start:end]
        
        err = preds - actuals
        loss = (err*err).sum()

        loss.backward()
        epoch_loss += loss

        optimizer.step(model)

    print(epoch, epoch_loss)

