from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch_ort
import torch_ort_optimizers

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        out = self.fc1(x)
        return out

input_size = 1
hidden_size = 1
num_classes = 1
batch_size = 1

def printBefore(model, lr):
    for p in model.parameters():
        print('data = ', p.data, 'grad=', p.grad, 'calculated=', p.data - (p.grad * lr))

def printAfter(model):
    cpulist = list(p.data.cpu() for p in model.parameters())

    print("Step on ORT Results")
    for c in cpulist:
        print("data=", c)

batch = torch.rand((batch_size, input_size))
y = torch.sin(batch)
model = NeuralNet(input_size, hidden_size, num_classes)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-1
# Forward pass: compute predicted y by passing x to the model.
y_pred = model(batch)

# Compute and print loss.
loss = loss_fn(y_pred, y)

# Backward pass: compute gradient of the loss with respect to model
# parameters
loss.backward()

print("Before step, learning_rate=", learning_rate, ":")
printBefore(model, learning_rate)

# step on ORT
model = model.to(torch.device('ort'))
optimizer = torch_ort_optimizers.SGD(model.parameters(), lr=learning_rate)
optimizer.step()

printAfter(model)
