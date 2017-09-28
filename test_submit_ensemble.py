import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from model.u_net import get_unet_1024, get_unet_1024_weighted, get_unet_1024_hq

import params

input_size = params.input_size
batch_size = params.batch_size
orig_width = params.orig_width
orig_height = params.orig_height
downscale = params.downscale
upscale = params.upscale
threshold = params.threshold

df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))

### Create bounds dictionary ###
test_bounds = pd.read_csv('input/test_bounds.csv')
test_bounds_dict = {}
for i, row in test_bounds.iterrows():
    test_bounds_dict[row['img'][:-4]] = (row['y_min'], row['y_max'], row['x_min'], row['x_max'])
### Create bounds dictionary ###


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


rles = []

model_1 = get_unet_1024_hq()
model_1.load_weights(filepath='weights/best_weights_1024x1024.hdf5')
model_2 = get_unet_1024_weighted()
model_2.load_weights(filepath='weights/best_weights_1024x1024_weighted.hdf5')
model_3 = get_unet_1024_hq()
model_3.load_weights(filepath='weights/best_weights_1024x1024_hq.hdf5')
models = [model_1, model_2, model_3]

models_weights = np.asarray([0.3, 0.1, 0.6], dtype=np.float32)

print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
for start in tqdm(range(0, len(ids_test), batch_size)):
    x_batch = []
    batch_names = []
    end = min(start + batch_size, len(ids_test))
    ids_test_batch = ids_test[start:end]

    for id in ids_test_batch.values:
        y_min, y_max, x_min, x_max = test_bounds_dict[id]
        img = cv2.imread('input/test_hq/{}.jpg'.format(id))
        img = img[y_min:y_max, x_min:x_max]
        img = cv2.resize(img, (input_size, input_size), None, 0, 0, downscale)
        x_batch.append(img)
        batch_names.append(id)

    x_batch = np.array(x_batch, np.float32) / 255

    ensemble_preds = np.zeros((len(models), len(x_batch), input_size, input_size))
    for model_idx in range(len(models)):
        preds = models[model_idx].predict_on_batch(x_batch)
        preds = np.squeeze(preds, axis=3)
        ensemble_preds[model_idx] = preds
    preds = np.dot(ensemble_preds.transpose(), models_weights)

    preds = np.zeros((len(x_batch), input_size, input_size), dtype=np.float32)
    for model_idx in range(len(ensemble_preds)):
        preds += ensemble_preds[model_idx] * models_weights[model_idx]


    for pred_num in range(len(preds)):
        pred = preds[pred_num]
        pred_name = batch_names[pred_num]
        y_min, y_max, x_min, x_max = test_bounds_dict[pred_name]

        prob = cv2.resize(pred, (x_max - x_min, y_max - y_min), None, 0, 0, upscale)
        mask = prob > threshold
        mask_full = np.zeros((orig_height, orig_width), dtype=np.int8)
        mask_full[y_min:y_max, x_min:x_max] = mask

        rle = run_length_encode(mask_full)
        rles.append(rle)

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')
