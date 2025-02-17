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


def bespoke_cov_matrix(z):
    """
    Parameters
    ----------
    z : list of tasks.  Each element of list should be an array of length M = (number of samples).

    Returns
    -------
    Cz : ndarray
        M x M empirical covariance matrix.
    """

    n = len(z)

    Cz = 0
    for i in range(n):
        Cz = Cz + z[i]@z[i].T
    Cz = Cz/(n)

    return Cz




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


def update_gaussian_cov_matrix(Cz, n, delta_n):
    M = np.shape(Cz)[0]
    new_cov_contrib = delta_n*gaussian_partitions_cov_matrix(M,delta_n)
    newCz = (1/(n + delta_n))*(n*Cz + new_cov_contrib)
    return newCz

def update_random_cov_matrix(Cz, n, delta_n):
    M = np.shape(Cz)[0]
    new_cov_contrib = delta_n*random_partitions_cov_matrix(M,delta_n)
    newCz = (1/(n + delta_n))*(n*Cz + new_cov_contrib)
    return newCz


class PartitionsCovMatrix:
    
    def __init__(self, M, n_initial, method = 'binary'):
        self.M = M
        self.n = n_initial
        self.method = method
        self.matrix = None

    def initialize_cov_matrix(self):
        if self.method == 'binary':
            Cz = random_partitions_cov_matrix(self.M, self.n)
        elif self.method == 'gaussian':
            Cz = gaussian_partitions_cov_matrix(self.M, self.n)
        else:
            raise ValueError(
                "method must be either 'binary' or 'gaussian'.")
        self.matrix = Cz
        
        return None

    def update_cov_matrix(self, add_n):
        if self.method == 'binary':
            Cz = update_random_cov_matrix(self.matrix, self.n, add_n)
        elif self.method == 'gaussian':
            Cz = update_gaussian_cov_matrix(self.matrix, self.n, add_n)
        else:
            raise ValueError(
                "method must be either 'binary' or 'gaussian'.")
        
        self.n = self.n + add_n
        self.matrix = Cz
        
        return None
    



# def get_Jacobian_matrices(model, inputs):
#     for i in range(inputs.shape[0]):
#         # TO DO
    
    