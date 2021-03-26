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

def printOptimizer(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue

        print('p.data=', p.data.cpu(), ', p.grad=', p.grad.cpu())
        
        state = optimizer.state[p]
        if len(state) != 0:            
            print('step=', state['step'].cpu(), ', exp_avg=', state['exp_avg'].cpu(), ', exp_avg_sq=', state['exp_avg_sq'].cpu())

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

# step on ORT
model = model.to(torch.device('ort'))
optimizer = torch_ort_optimizers.AdamW(model.parameters(), lr=learning_rate)
print("Before step, learning_rate=", learning_rate, ":")
printOptimizer(optimizer)

optimizer.step()

print("After step #1:")
printOptimizer(optimizer)

optimizer.step()
print("After step #2:")
printOptimizer(optimizer)
