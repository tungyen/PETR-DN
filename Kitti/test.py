import os
import numpy as np
import torch

a = torch.zeros(20, 7)
b = torch.ones(3, 7)
a[:b.shape[0], :] = b