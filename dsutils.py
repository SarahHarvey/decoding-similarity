"""
Miscellaneous helper functions.
"""

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

class SoftMaxModule(nn.Module):
    def __init__(self):
        super(SoftMaxModule, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, x):
        return self.softmax(x)

# def get_Jacobian_matrices(model, inputs):
#     for i in range(inputs.shape[0]):
#         # TO DO
    
    