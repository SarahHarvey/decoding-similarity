"""
Jacobian similarity helper functions.
"""

import torch
import warnings
import numpy as np
import metrics

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



def compute_Jacobian_Bures_distances(J_dict, model_names):
    """
    Computes a matrix of Bures distances between the Jacobians given in a dictionary J_dict  

    Parameters
    ----------
    J_dict : Dictionary of Jacobians associated with each model.  Keys are model names.  Each value is a list of torch tensors extracted by the function decoding_jacobian().    

    model_names:  list of strings.
        List of models to compute the distance between, and the order they will be listed in in the distance matrix.

    Returns
    -------
    bures_dists_all : list
        List of bures distance matrices.  The ith distance matrix is the distances among the models' ith Jacobians in their respective lists of Jacobians.
    """
    bures_dists_all = []
    
    for outer_ind in range(len(J_dict[model_names[0]])):
        bures_dists =  np.zeros((len(model_names),len(model_names)))
    
        for i in range(len(model_names)):
            for j in range(len(model_names)):
                if j < i:
                    JX = J_dict[model_names[i]][outer_ind].numpy()
                    JY = J_dict[model_names[j]][outer_ind].numpy()
                    bures_dists[i,j] = metrics.sq_bures_metric(JX,JY)
    
        bures_dists = bures_dists + bures_dists.T
        bures_dists_all_rbyn.append(bures_dists)
    
        print(str(outer_ind) + "/" + str(len(J_dict[model_names[0]])) )

    return bures_dists_all

