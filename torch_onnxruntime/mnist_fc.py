from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch_ort

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

#todo: move it to torch_ort as util function
def get_ort_device(devkind, index = 0):
    return torch.device("ort", torch_ort.get_ort_device(devkind, index))

input_size = 784
hidden_size = 500
num_classes = 10
batch_size = 128
model = NeuralNet(input_size, hidden_size, num_classes)

batch = torch.rand((batch_size, input_size))
with torch.no_grad():
    pred = model(batch)
    print("inference result is: ")
    print(pred)

    device = get_ort_device("Apollo")
    model.to(device)

    ort_batch = batch.to(device)
    ort_pred = model(ort_batch)
    print("ORT inference result is:")
    print(ort_pred.cpu())
    print("Compare result:")
    print(torch.allclose(pred, ort_pred.cpu(), atol=1e-6))
