import FizzBuzz
from FizzBuzzModel import FizzBuzzModel
import torch
import torch.nn as nn
import numpy as np

ENABLE_GPU = torch.cuda.is_available()

NUM_DIGIT_BINARY = 10
NUM_HIDDEN = 100
NUM_OUT = 4


def input_encoder(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])


x_train = torch.Tensor([input_encoder(i, NUM_DIGIT_BINARY) for i in range(101, 2 ** NUM_DIGIT_BINARY)])
y_lables = torch.LongTensor([FizzBuzz.fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGIT_BINARY)])
model = FizzBuzzModel(NUM_DIGIT_BINARY, NUM_HIDDEN, NUM_OUT)

if ENABLE_GPU:
    device = torch.device("cuda")
    x_train = x_train.to(device)
    y_lables = y_lables.to(device)
    model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

BATCH_SIZE = 128
for epoch in range(2500):
    for start in range(0, len(x_train), BATCH_SIZE):
        end = start + BATCH_SIZE
        X_train = x_train[start:end]
        Y_lables = y_lables[start:end]

        Y_pre = model(X_train)
        loss = loss_fn(Y_pre, Y_lables)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_p = loss_fn(model(X_train), Y_lables).item()
        print('Epoch:', epoch, 'Loss:', loss_p)
