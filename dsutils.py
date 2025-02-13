"""
Miscellaneous helper functions.
"""

import torch
import warnings
import numpy as np

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


def random_partitions_cov_matrix(M, n):
    """
    Parameters
    ----------
    M : int (number of samples)
    n : int (number of random tasks to generate)

    Returns
    -------
    Cz : ndarray
        M x M empirical covariance matrix.
    """
    z = []
    for i in range(n):
        zrand = np.random.randint(0,2,(M,1)) 
        zrand = 2*zrand - 1
        z.append(zrand)

    Cz = 0
    for i in range(n):
        Cz = Cz + z[i]@z[i].T
    Cz = Cz/(n)

    return Cz


def gaussian_partitions_cov_matrix(M, n):
    """
    Parameters
    ----------
    M : int (number of samples)
    n : int (number of random tasks to generate)

    Returns
    -------
    Cz : ndarray
        M x M empirical covariance matrix.
    """
    z = []
    for i in range(n):
        zrand = np.random.normal(0,1,(M,1)) 
        z.append(zrand)

    Cz = 0
    for i in range(n):
        Cz = Cz + z[i]@z[i].T
    Cz = Cz/(n)

    return Cz



# def get_Jacobian_matrices(model, inputs):
#     for i in range(inputs.shape[0]):
#         # TO DO
    
    