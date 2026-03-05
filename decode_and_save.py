import argparse
import pickle
import numpy as np
import dsutils
from sklearn.preprocessing import StandardScaler
from itertools import product

np.random.seed(41)

parser = argparse.ArgumentParser(description='Decode representations from deep nets using MLP probes')
parser.add_argument('--model_name', type=str, default='resnet50', help='Name of model to decode representations from')
parser.add_argument('--dim_reduction_method', type=str, default='RP', help='Dimensionality reduction method to apply to representations before decoding (PCA, RP, or None)')
parser.add_argument('--P', type=int, default=5001, help='Number of dimensions to project to if using RP, or percentage of variance to preserve if using PCA (e.g. 0.98 for 98%)')
parser.add_argument('--image_sample', type=str, default='imagenet_random_sample_5000_v1', help='Name of imagenet sample used to extract representations')
parser.add_argument('--test_set_size', type=int, default=1000)
parser.add_argument('--kernel', type=str, default='linear')
# parser.add_argument('--eval_on_test_set', action='store_true', help='Whether to evaluate decoding performance on a held-out test set (instead of training set)')
args = parser.parse_args()

dim_reduction_method = args.dim_reduction_method  # 'PCA' or 'RP'
P = args.P
model_name = args.model_name
image_sample = args.image_sample
test_set_size = args.test_set_size
kernel = args.kernel
eval_on_test_set = True #args.eval_on_test_set


if dim_reduction_method == 'RP':
    # TODO: Check if the RP-reduced representations exist, if not, run RP_reps.py to create them
    with open (f'reps_rps/internal_reps_rps_{model_name}_{image_sample}_RPto{P}dims.pkl', 'rb') as fp:
        repDict = pickle.load(fp)
if dim_reduction_method == 'PCA':
    with open (f'reps_pca/internal_reps_pca_{model_name}_{image_sample}_PCA{P*100}percent.pkl', 'rb') as fp:
        repDict = pickle.load(fp)
if dim_reduction_method is None:
    with open (f'reps/{model_name}_{image_sample}.pkl', 'rb') as fp:
        repDict = pickle.load(fp)
if dim_reduction_method not in ['PCA', 'RP', None]:
    raise ValueError("Invalid dim_reduction_method. Must be 'PCA' or 'RP' or None")

internal_reps = [rep for rep in repDict.values()]
layer_names = [name for name in repDict.keys()]

print(model_name + "_" + "_".join(layer_names))

# TODO:  Improve target loading!!! 

# Zfull = np.load(f'targets/C_{image_sample}.npy')
with open('algonauts_brain_data_joint_images_8subjects.pkl', 'rb') as f:
    brainData = pickle.load(f)

# dict_keys(['lh_all', 'rh_all', 'lh_V1v', 'lh_V1d', 'lh_V2v', 'lh_V2d', 'lh_V3v', 'lh_V3d', 'lh_hV4', 'lh_EBA', 'lh_FBA-1', 'lh_FBA-2', 'lh_mTL-bodies', 'lh_OFA', 'lh_FFA-1', 'lh_FFA-2', 'lh_mTL-faces', 'lh_aTL-faces', 'lh_OPA', 'lh_PPA', 'lh_RSC', 'lh_OWFA', 'lh_VWFA-1', 'lh_VWFA-2', 'lh_mfs-words', 'lh_mTL-words', 'lh_early', 'lh_midventral', 'lh_midlateral', 'lh_midparietal', 'lh_ventral', 'lh_lateral', 'lh_parietal', 'lh_all-prf-visual', 'lh_all-bodies', 'lh_all-faces', 'lh_all-places', 'lh_all-words', 'lh_all-streams', 'rh_V1v', 'rh_V1d', 'rh_V2v', 'rh_V2d', 'rh_V3v', 'rh_V3d', 'rh_hV4', 'rh_EBA', 'rh_FBA-1', 'rh_FBA-2', 'rh_mTL-bodies', 'rh_OFA', 'rh_FFA-1', 'rh_FFA-2', 'rh_mTL-faces', 'rh_aTL-faces', 'rh_OPA', 'rh_PPA', 'rh_RSC', 'rh_OWFA', 'rh_VWFA-1', 'rh_VWFA-2', 'rh_mfs-words', 'rh_mTL-words', 'rh_early', 'rh_midventral', 'rh_midlateral', 'rh_midparietal', 'rh_ventral', 'rh_lateral', 'rh_parietal', 'rh_all-prf-visual', 'rh_all-bodies', 'rh_all-faces', 'rh_all-places', 'rh_all-words', 'rh_all-streams'])
region = 'lh_V1v'
Zfull = brainData[0][region] 

for i in range(len(brainData)):
    internal_reps.append(brainData[i]['lh_all'])
    layer_names.append(f'Subject_{i+1}')    

#### Loop over all parameter combos (no cross-validation) and save all predictions

# Make sure test_indices doesn't exceed the size of your data
max_samples = internal_reps[0].shape[0]
test_set_size = min(test_set_size, max_samples)
test_indices = np.random.choice(np.arange(max_samples), test_set_size, replace=False)

if kernel == 'linear':
    param_grid = {
        'a': [1],
        'b' : list(np.logspace(-6, 7, num=20, endpoint=True, base=10.0, dtype=None, axis=0)),
        'gamma': [1]
    }
elif kernel == 'rbf':
    param_grid = {
        'a': [1],
        'b' : list(np.logspace(-8, 3, num=8, endpoint=True, base=10.0, dtype=None, axis=0)),
        'gamma': list(np.logspace(-8, 3, num=8, endpoint=True, base=10.0, dtype=None, axis=0))
    }

# Create all parameter combinations
param_keys = list(param_grid.keys())
param_combos = [dict(zip(param_keys, vals)) for vals in product(*param_grid.values())]

# resultDict_all: stores results for every (layer, param_combo) pair
resultDict_all = {}
resultDict_all["layer_names"] = layer_names
resultDict_all["param_combos"] = param_combos
resultDict_all["Z"] = Zfull
# results[layer_idx] is a list of dicts, one per param combo
resultDict_all["results"] = []
resultDict_all["kernel"] = kernel
resultDict_all["model_name"] = model_name
resultDict_all["image_sample"] = image_sample

for i in range(len(internal_reps)):
    layer_results = []

    # Split data into train and test
    if eval_on_test_set:
        Z_test = Zfull[test_indices]
        X_full = internal_reps[i]
        X_test = X_full[test_indices]
        X = np.delete(X_full, test_indices, axis=0)
        Z = np.delete(Zfull, test_indices, axis=0)
    else:
        Z_test = Zfull
        Z = Zfull
        X_full = internal_reps[i]
        X = X_full
        X_test = X_full

    # Fit scaler on training data only and transform both train and test
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    M = X.shape[0]

    for p_idx, params in enumerate(param_combos):
        probe = dsutils.genKernelRegression(
            center_columns=True, kernel=kernel,
            a=params['a'], b=params['b'], gamma=params['gamma'],
            fit_intercept=False
        )
        probe.fit(X, Z)

        Z_pred = probe.predict(X_test)
        R2 = probe.score(X_test, Z_test)
        df = probe.effective_dof()
        norm_sq = probe.rkhs_norm()

        layer_results.append({
            'params': params,
            'Z_pred': Z_pred,
            'R2': R2,
            'X_train': X,
            'Z_train': Z,
            'X_test': X_test,
            'Z_test': Z_test,
            'effective_dof': df,
            'rkhs_norm': norm_sq
        })

        print(f"  Layer {layer_names[i]} | params {params} | R²={R2:.4f}")

    resultDict_all["results"].append(layer_results)
    print(f"--- {layer_names[i]}  {i+1}/{len(internal_reps)} done ---\n")

print("All done.")

with open(f'decoding_results/NSD_decoding_subj0_{region}_{model_name}_{kernel}_kernel_{image_sample}_RPto{P}dims.pkl', 'wb') as fp:
    pickle.dump(resultDict_all, fp)