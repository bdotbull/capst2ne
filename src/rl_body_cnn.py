
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
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

def pre_process():
    pass

dataset = tf.data.TFRecordDataset("../temp/*.tfrecord")
dataset = dataset.map(pre_process)
dataset = dataset.snapshot("/path/to/snapshot_dir")
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.batch(batch_size=32)

dataset = dataset.prefetch()  # not sure if necessary.  Files are ~1.0mb-2.0mb
                              # this would be great for files 10mb-100mb


# Model Bits n Bobs
dense_layers = [2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

# Model Building
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(
                conv_layer, layer_size, dense_layer, int(time.tim()))
            tensorboard = TensorBoard(log_dir=f'logs/{int(time.time())}')
            pass