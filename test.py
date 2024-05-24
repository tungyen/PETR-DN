import torch

a = torch.randn((20, 8))
b = torch.max(a, dim=1)
print(b)