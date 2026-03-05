# Extract representations resulting from probe inputs image_data for a list of models, and save them individually (takes a long time)

import argparse

import extract_internal_reps
import pickle
import numpy as np

parser = argparse.ArgumentParser(description='Extract and save representations from deep nets for probe inputs')
parser.add_argument('--model_name', type=str, default='resnet50', help='Name of model to extract representations from')
parser.add_argument('--image_sample', type=str, default='imagenet_random_sample_5000_v1', help='Name of image sample used to extract representations')
args = parser.parse_args()

if args.image_sample == 'algonauts_joint_images_8subjects':
    shared_images = np.load('algonauts_joint_images_8subjects.npy')
else:
    with open('imagenet_probe_images/' + args.image_sample + '.pkl', 'rb') as f:
        image_data = pickle.load(f)
        shared_images = image_data['images']

repDict = {}

model_name = args.model_name #["dinov2_vitb14"] #, "inception_v3", "vit_b_16", "swin_v2_t"]
weights = 'first'
image_data = shared_images
batch_size = 32

repDict[model_name] = extract_internal_reps.get_model_activations(model_name, weights, image_data, batch_size=32, saverep = True, filename = args.image_sample)
print(model_name + " done")