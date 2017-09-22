import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from deterministic_augmentation import deterministic_augmentation

import params

input_size = params.input_size
batch_size = params.batch_size
orig_width = params.orig_width
orig_height = params.orig_height
downscale = params.downscale
upscale = params.upscale
threshold = params.threshold
model = params.model_factory()

df_train = pd.read_csv('input/train_masks.csv')
ids_test = df_train['img'].map(lambda s: s.split('.')[0])

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))

### Create bounds dictionary ###
test_bounds = pd.read_csv('input/train_bounds.csv')
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
for start in tqdm(range(0, len(ids_test))):
    x_batch = []
    batch_names = []
    id = ids_test[start]

    y_min, y_max, x_min, x_max = test_bounds_dict[id]
    img = cv2.imread('input/train/{}.jpg'.format(id))
    img = img[y_min:y_max, x_min:x_max]
    img = cv2.resize(img, (input_size, input_size), None, 0, 0, downscale) / 255.0
    batch_names.append(id)
 
    transform_params = []
    # transform_params.append((0.0, 0, 0, False, False, False)) # original image is not augmented
    transform_params.append(False) # original image is not augmented
    x_batch.append(img)

    for augment in range(batch_size - 1):
        # theta = np.random.uniform(-20.0, 20.0)
        # wshift = np.random.uniform(-0.1, 0.1)
        # hshift = np.random.uniform(-0.1, 0.1)
        # u_rotate = np.random.random() < 0.5
        u_flip = np.random.random() < 0.5
        # u_shift = np.random.random() < 0.3
        # x_batch.append(deterministic_augmentation(img, theta, wshift, hshift, u_rotate, u_flip, u_shift, image = True))
        x_batch.append(deterministic_augmentation(img, u_flip, image = True))
        # transform_params.append((theta, wshift, hshift, u_rotate, u_flip, u_shift))
        transform_params.append(u_flip)
    x_batch = np.array(x_batch, np.float32)
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    
    for i in range(preds.shape[0]):
        # theta, wshift, hshift, u_rotate, u_flip, u_shift = transform_params[i]
        # preds[i, ...] = deterministic_augmentation(preds[i, ...], -theta, -wshift, -hshift, u_rotate, u_flip, u_shift, image = False)
        u_flip = transform_params[i]
        preds[i, ...] = deterministic_augmentation(preds[i, ...], u_flip, image = False)
    pred = np.zeros((preds.shape[1], preds.shape[2]))
    for i in range(preds.shape[0]):
        pred += preds[i, ...]

    pred /= float(batch_size)

    pred_name = batch_names[0]
    y_min, y_max, x_min, x_max = test_bounds_dict[pred_name]
    prob = cv2.resize(pred, (x_max - x_min, y_max - y_min), None, 0, 0, upscale)
    prob = gaussian_filter(prob, sigma=2)
    mask = prob > threshold
    mask_full = np.zeros((orig_height, orig_width), dtype=np.int8)
    mask_full[y_min:y_max, x_min:x_max] = mask

    rle = run_length_encode(mask_full)
    rles.append(rle)


print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/submission_train_1024_augm.csv.gz', index=False, compression='gzip')
