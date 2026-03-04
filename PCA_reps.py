# Load some saved representations and use PCA to find projection that preserves a given % of variance, 
# for each layer of a model. Save the PCA projections for later use in decoding analyses.

import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
import gc

# Base directory of this script, used for resolving file paths
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def pca_reps(model_name, imgsample, var_threshold=0.98):
    """Apply PCA to each layer's representations, keeping enough components
    to explain `var_threshold` fraction of the variance.

    Parameters
    ----------
    model_name : str
        Name of the model whose representations to load.
    imgsample : str
        Name of the image sample (used in filenames).
    var_threshold : float
        Fraction of variance to retain (e.g. 0.98 for 98%).

    Returns
    -------
    repDict_pca : dict
        Mapping from layer name to PCA-projected representation array.
    """
    reps_path = os.path.join(_BASE_DIR, 'reps', f'{model_name}_{imgsample}.pkl')
    with open(reps_path, 'rb') as f:
        repDict = pickle.load(f)
        print(model_name)

    layer_names = list(repDict[model_name].keys())

    repDict_pca = {}

    for idx, layer_name in enumerate(layer_names):
        rep = repDict[model_name][layer_name]
        rep = rep.reshape(rep.shape[0], -1)
        del repDict[model_name][layer_name]
        gc.collect()

        pca = PCA(n_components=var_threshold, svd_solver='full')
        pcs = pca.fit_transform(rep)
        repDict_pca[layer_name] = pcs
        print(f'Layer {idx} ({layer_name}): explained variance = {np.sum(pca.explained_variance_ratio_):.4f}, '
              f'n_components = {pca.n_components_}')
        del rep, pcs, pca
        gc.collect()

    out_dir = os.path.join(_BASE_DIR, 'reps_pca')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'internal_reps_pca_{model_name}_{imgsample}_PCA{var_threshold*100:.0f}percent.pkl')
    with open(out_path, 'wb') as fp:
        pickle.dump(repDict_pca, fp)

    return repDict_pca


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PCA-project model representations to preserve a fraction of variance.')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Name of model')
    parser.add_argument('--imagenet_sample', type=str, default='imagenet_random_sample_5000_v5', help='Name of imagenet sample')
    parser.add_argument('--var_threshold', type=float, default=0.98, help='Fraction of variance to retain (e.g. 0.98)')
    args = parser.parse_args()
    pca_reps(args.model_name, args.imagenet_sample, args.var_threshold)