
"""
import torch

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
"""

import time
import os
import sys
import pathlib
import PIL
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.engine.sequential import Sequential


#### Come back to this block for CapsTHREE if still using TF
def pre_process():
    pass

dataset = tf.data.TFRecordDataset("../temp/*.tfrecord")
dataset = dataset.map(pre_process)
dataset = dataset.snapshot("/path/to/snapshot_dir")
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.batch(batch_size=32)

dataset = dataset.prefetch()  # not sure if necessary.  Files are ~1.0mb-2.0mb
                              # this would be great for files 10mb-100mb
#### Come back to this block for CapsTHREE if still using TF

# DATA LOADER
# TODO: set up Data directory to have dominus, fennec, 
#       and octane as the top level
#data_dir = pathlib.Path('../data/img/')

# LOADER PARAMETERS
batch_size = 32
img_height = 180
img_width = 180

num_classes = 3  #=len(train_ds.class_names).  Used for last Dense

# Model Bits n Bobs
dense_layers = [1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

# Model Building
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(
                conv_layer, layer_size, dense_layer, int(time.tim()))
            tensorboard = TensorBoard(log_dir=f'logs/{int(time.time())}')
            print(NAME)
            model = Sequential()

            model.add(preprocessing.Rescaling(1./255,
                input_shape=(img_height, img_width, 3)))
            model.add(Conv2D(layer_size, 3, padding='same'))
            model.add(Activation('relu'))
            model.add(MaxPooling2D())  # maybe try 2? pool_size=(2, 2)

            # LOOP range(conv_layer-1) because we have one already
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, 3, padding='same'))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))  # maybe 2 ??

            model.add(Flatten())    # Flatten before any Dense

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(num_classes))
            model.add(Activation('relu'))

            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
