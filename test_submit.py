import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter

import params

input_size = params.input_size
batch_size = params.batch_size
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model = params.model_factory()

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

model.load_weights(filepath='weights/best_weights.hdf5')

print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
for start in tqdm(range(0, len(ids_test), batch_size)):
    x_batch = []
    batch_names = []
    end = min(start + batch_size, len(ids_test))
    ids_test_batch = ids_test[start:end]

    for id in ids_test_batch.values:
        y_min, y_max, x_min, x_max = test_bounds_dict[id]
        img = cv2.imread('input/test/{}.jpg'.format(id))
        img = img[y_min:y_max, x_min:x_max]
        img = cv2.resize(img, (input_size, input_size))
        x_batch.append(img)
        batch_names.append(id)

    x_batch = np.array(x_batch, np.float32) / 255
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)

    for pred_num in range(len(preds)):
        pred = preds[pred_num]
        pred_name = batch_names[pred_num]
        y_min, y_max, x_min, x_max = test_bounds_dict[pred_name]

        prob = cv2.resize(pred, (x_max - x_min, y_max - y_min))
        prob = gaussian_filter(prob, sigma=2)
        mask = prob > threshold
        mask_full = np.zeros((orig_height, orig_width), dtype=np.int8)
        mask_full[y_min:y_max, x_min:x_max] = mask

        rle = run_length_encode(mask_full)
        rles.append(rle)

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')
