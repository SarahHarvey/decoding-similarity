import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.random_projection import SparseRandomProjection
import gc

# Base directory of this script, used for resolving file paths
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def random_proj_reps(model_name, imgsample, P):
    
    reps_path = os.path.join(_BASE_DIR, 'reps', model_name + '_' + imgsample + '.pkl')
    with open(reps_path, 'rb') as f:
        repDict = pickle.load(f)
        print(model_name)

    layer_names = list(repDict[model_name].keys())
    
    repDict_rps = {}

    for idx, layer_name in enumerate(tqdm(layer_names)):
        # Load one layer at a time, flatten, then remove from repDict to free memory
        rep = repDict[model_name][layer_name]
        rep = rep.reshape(rep.shape[0], -1)
        del repDict[model_name][layer_name]
        gc.collect()

        n_samples, n_features = rep.shape
        # Skip projection if the layer is already smaller than P
        if n_features <= P:
            print(f"Layer {idx} ({n_features}d) — kept as-is (already <= {P}d)")
            repDict_rps[layer_name] = rep
            continue

        # SparseRandomProjection builds a sparse CSR matrix internally
        projector = SparseRandomProjection(n_components=P, random_state=42, dense_output=True)
        projected = projector.fit_transform(rep)
        print(f"Layer {idx} ({n_features}d -> {P}d) done, projected shape: {projected.shape}")
        repDict_rps[layer_name] = projected
        del rep, projected, projector
        gc.collect()

    out_dir = os.path.join(_BASE_DIR, 'reps_rps')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'internal_reps_rps_{model_name}_{imgsample}_RPto{P}dims.pkl')
    with open(out_path, 'wb') as fp:
        pickle.dump(repDict_rps, fp)

    return repDict_rps

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Randomly project model representations to a lower-dimensional space.')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Name of model to project representations for')
    parser.add_argument('--image_sample', type=str, default='imagenet_random_sample_5000_v5', help='Name of imagenet sample')
    parser.add_argument('--P', type=int, default=5001, help='Number of dimensions to project to')
    args = parser.parse_args()
    random_proj_reps(args.model_name, args.image_sample, args.P)