import FizzBuzz
import torch
import torch.nn as nn
import numpy as np


class FizzBuzzModel(torch.nn.Module):
    def __init__(self, x_in, h, y_out):
        super(FizzBuzzModel, self).__init__()
        self.linear1 = nn.Linear(x_in, h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(h, y_out)

    def forward(self, x_train):
        h_relu = self.linear1(x_train).relu()
        y_trian = self.linear2(h_relu)

        return y_trian
