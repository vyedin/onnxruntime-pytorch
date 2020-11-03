import torch

device = torch.device("ort")
x = torch.zeros(5, 3, device = device)
print(x)
