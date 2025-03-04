"""
Jacobian similarity helper functions.
"""

import torch
import warnings
import numpy as np


def decoding_jacobian(x1, model_2nd_half):
    Js = []
    for i in np.arange(x1.shape[0]):
        decode_jacobian = torch.autograd.functional.jacobian(model_2nd_half, x1[i,:])
        Js.append(decode_jacobian)
        print(i)
    return Js


def convert_Jacobian(J_dict, model_name):
    
    Js = []
    
    Ncats = J_dict[model_name][0].shape[0]
    Nimgs = len(J_dict[model_name])
    
    for j in range(Ncats):
    
        J_mbyn = J_dict[model_name][0][j,:].numpy()
        J_mbyn.reshape((1,J_mbyn.shape[0]))
        
        for i in np.arange(1,Nimgs):
            newrow = J_dict[model_name][i][j,:].numpy()
            J_mbyn = np.vstack([J_mbyn, newrow])
        
        Js.append(J_mbyn)
        print(j)
    return Js