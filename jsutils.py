"""
Jacobian similarity helper functions.
"""

import torch
import warnings
import numpy as np


def decoding_jacobian(x1, model_2nd_half):
    Js = []
    for i in np.arange(50):#np.arange(x1.shape[0]):
        decode_jacobian = torch.autograd.functional.jacobian(model_2nd_half, x1[i,:])
        Js.append(decode_jacobian)
        print(i)
    return Js