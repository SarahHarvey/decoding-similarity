"""
Miscellaneous helper functions.
"""

import matplotlib.pyplot as plt
import torch
import warnings

from torch import nn
from torchvision import models

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


# class SoftMax(nn.Module):
#     def __init__(self, x):
#         super().__init__()
#         self.softmax = torch.nn.functional.softmax(x, dim=1)

#     def forward(self, x):
#         return self.softmax(x, dim=1)


class SoftMaxModule(nn.Module):
    def __init__(self):
        super(SoftMaxModule, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, x):
        return self.softmax(x)