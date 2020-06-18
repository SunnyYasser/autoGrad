import numpy as np
from typing import List
from autograd.tensor import Tensor
from autograd.parameters import Parameter
from autograd.module import Module
from autograd.optim import SGD
from autograd.activation import * 

def binary_encode(x: int) -> List[int]:
    return [x >> i & 1 for i in range(10)]

def fizz_encode(x : int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0 ,0 ,0]


x_train = Tensor([binary_encode(x) for x in range(101, 1024)])
y_train = Tensor([fizz_encode(x) for x in range(101, 1024)])
    
class Simple_Model(Module):
    def __init__(self, num_hidden:int = 50) -> None:
        self.w1  = Parameter(10,num_hidden)
        self.b1  = Parameter(num_hidden)
        
        self.w2 = Parameter(num_hidden,4)
        self.b2 = Parameter(4)

    def predict(self, x:Tensor) -> Tensor:
        # x -> (batch_size, 10)
        
        x1 = x @ self.w1 + self.b1 #(batch_size, hidden)
        x2 = sigmoid(x1) #(batch_size, hidden)

        x3 = x2 @ self.w2 + self.b2 #(batch_Size, 4)
        
        return x3

model = Simple_Model()    
lr = 0.001
batch_size = 32
optimizer = SGD(lr)

starts = np.arange(0, x_train.shape[0], batch_size)
for epoch in range(100):
    
    epoch_loss = 0.0
    np.random.shuffle(starts)

    for start in starts:
        # print(start)
        
        end = start + batch_size
        inputs = x_train[start:end]
        model.zero_grad()

        preds = model.predict(inputs)
        actuals = y_train[start:end]
        
        err = preds - actuals
        loss = (err*err).sum()

        loss.backward()
        epoch_loss += loss

        optimizer.step(model)

    print(epoch, epoch_loss)


correct_count = 0
for x in range(1, 101):
    inputs = Tensor(binary_encode(x))
    preds = model.predict(inputs)[0]
    pred_idx = np.argmax(preds)
    actual_idx = np.argmax(fizz_encode(x))

    labels = [str(x), "fizz", "buzz", "fizzbuzz"]

    if pred_idx == actual_idx:
        correct_count += 1
    
    print(x, labels[pred_idx], labels[actual_idx])
print("Correct : ", correct_count)

