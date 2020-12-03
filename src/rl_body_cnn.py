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

"""
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
"""
# limit gpu mem usage to 33%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# DATA LOADER
# for initial testing, using the greybox images
data_dir = pathlib.Path('../data/img/screenshots/train/cropped/')
# Set up classes
dominus = list(data_dir.glob('dominus/*'))
fennec = list(data_dir.glob('fennec/*'))
octane = list(data_dir.glob('octane/*')))

# LOADER PARAMETERS
batch_size = 32
img_height = 180
img_width = 180

# 80% of images used for training, leaving 20% for validation
# Note: `.image_dataset_from_directory` resizes the images.
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=327,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# create 20% validation set
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=327,
    image_size=(img_height, img_width),
    batch_size=batch_size)

num_classes = len(train_ds.class_names)  #Used for last Dense

# Model Bits n Bobs
epochs = 10
dense_layers = [1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

# Model Building
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(
                conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir=f'../logs/{NAME}')
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

            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=[tensorboard]
            )
