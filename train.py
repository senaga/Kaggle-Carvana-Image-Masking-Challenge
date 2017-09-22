import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split

import params
import augmentation

input_size = params.input_size
downscale = params.downscale
epochs = params.max_epochs
batch_size = params.batch_size
model = params.model_factory()

df_train = pd.read_csv('input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])
ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

### Create bounds dictionary ###
train_bounds = pd.read_csv('input/train_bounds.csv')
train_bounds_dict = {}
for i, row in train_bounds.iterrows():
    train_bounds_dict[row['img'][:-4]] = (row['y_min'], row['y_max'], row['x_min'], row['x_max'])
### Create bounds dictionary ###

print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))


def get_image_and_mask(img_id, bounds=(0, params.orig_height, 0, params.orig_width)):
    y_min, y_max, x_min, x_max = bounds

    img = cv2.imread('input/train_hq/{}.jpg'.format(img_id))
    img = img[y_min:y_max, x_min:x_max]
    img = cv2.resize(img, (input_size, input_size), None, 0, 0, downscale)

    mask = Image.open('input/train_masks/{}_mask.gif'.format(img_id))
    mask = np.asarray(mask) * 255
    mask = mask[y_min:y_max, x_min:x_max]
    mask = cv2.resize(mask, (input_size, input_size), None, 0, 0, downscale)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    img, mask = img / 255., mask / 255.
    return img, mask


def train_generator():
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch.values:
                img, mask = get_image_and_mask(id, train_bounds_dict[id])
                img, mask = augmentation.random_augmentation(img, mask)
                mask = mask[:, :, 0]
                mask = np.expand_dims(mask, axis=2)

                x_batch.append(img)
                y_batch.append(mask)

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            yield x_batch, y_batch


def valid_generator():
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch.values:
                img, mask = get_image_and_mask(id, train_bounds_dict[id])
                mask = mask[:, :, 0]
                mask = np.expand_dims(mask, axis=2)

                x_batch.append(img)
                y_batch.append(mask)

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            yield x_batch, y_batch


callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4,
                               mode='min'),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min')]

# model.load_weights('weights/best_weights.hdf5')

model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
