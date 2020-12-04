"""
CNN helper functions for loading data
"""

import PIL
import matplotlib.pyplot as plt
#import cv2
import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def crop_one_img(img_path, out_dir='', car='unspecified'):
    """
    Crop an image to contain mainly car features

    Args:
        img_path (str or Path): input path
        out_dir (str, optional): desired output path. Defaults to '' and changed later.
        car (str, optional): label for car being processed. Defaults to 'unspecified' for labeling purposes.
    """
    #img = Image.open(image_path)
    #width, height = img.size
    
    # if no directory specified, put in cropped folder
    if not out_dir:
        out_dir = '../data/img/screenshots/train/cropped/'
    
    try:
        img = PIL.Image.open(img_path)
        #plt.imshow(img)

        # TODO: Make this so I can input any resolution
        # Crop the image using crop() method
        # (left, upper, right, lower)
        # at 2560x1080
        # left, upper = 0, 0
        # right, lower = (2560, 1080)
        box = (960, 500, 
              1600, 1000)
        croppedImage = img.crop(box)
        filename = img.filename.split('/')[-1]    # get just filename (not path)
        croppedImage.save(f'{out_dir}{car}/cropped_{filename}')
        #plt.imshow(croppedImage)
    
    except FileNotFoundError:
        print('Provided image path is not found')
        print(f'out_dir: {out_dir}')
        print(f'car: {car}')
        #print(f'input img.filename: {img.filename}')


def data_loader(input_path, class_names, batch_size=32,
    img_height=180, img_width=180):
    """[summary]

    Args:
        input_path (str): data directory
        class_names (list): list of labels corresponding to
                        directory names
        batch_size (int): [description]. Defaults to 32.
        img_height (int): [description]. Defaults to 180.
        img_width (int): [description]. Defaults to 180.
    """

    pass

def viz_res(history, epochs):
    """[summary]

    Args:
        history ([type]): [description]
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def plot_digit_weights(ax, weight, digit, img_height, img_width):
    """Plot the weights from fit cnn as image."""
    digit_weigths = np.reshape(weight[:,digit], (img_height, img_width))
    ax.imshow(digit_weigths, cmap=plt.cm.winter, interpolation="nearest") 

def viz_filters(model):
    #Iterate thru all the layers of the model
    for layer in model.layers:
        if 'conv' in layer.name:
            weights, bias= layer.get_weights()
            print(layer.name, filters.shape)
            
            #normalize filter values between  0 and 1 for visualization
            f_min, f_max = weights.min(), weights.max()
            filters = (weights - f_min) / (f_max - f_min)  
            print(filters.shape[3])
            filter_cnt=1
            
            #plotting all the filters
            for i in range(filters.shape[3]):
                #get the filters
                filt=filters[:,:,:, i]
                #plotting each of the channel, color image RGB channels
                for j in range(filters.shape[0]):
                    ax= plt.subplot(filters.shape[3], filters.shape[0], filter_cnt  )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.imshow(filt[:,:, j])
                    filter_cnt+=1
            plt.show()


def viz_feature_map(model, img_path):
    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after
    # the first.
    successive_outputs = [layer.output for layer in model.layers[1:]]

    #visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

    # Let's prepare a random input image of a cat or dog from the training set.
    #cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
    #dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]

    #img_path = random.choice(cat_img_files + dog_img_files)

    img = load_img(img_path, target_size=(150, 150))  # this is a PIL image

    x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
    x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

    # Rescale by 1/255
    x /= 255.0

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]

    # -----------------------------------------------------------------------
    # Now let's display our representations
    # -----------------------------------------------------------------------
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
      print(feature_map.shape)
      if len(feature_map.shape) == 4:
          
        #-------------------------------------------
        # Just do this for the conv / maxpool layers, not the fully-connected layers
        #-------------------------------------------
        n_features = feature_map.shape[-1]  # number of features in the feature map
        size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)

        # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))

        #-------------------------------------------------
        # Postprocess the feature to be visually palatable
        #-------------------------------------------------
        for i in range(n_features):
          x  = feature_map[0, :, :, i]
          x -= x.mean()
          x /= x.std ()
          x *=  64
          x += 128
          x  = np.clip(x, 0, 255).astype('uint8')
          display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

        #-----------------
        # Display the grid
        #-----------------

        scale = 20. / n_features
        plt.figure( figsize=(scale * n_features, scale) )
        plt.title ( layer_name )
        plt.grid  ( False )
        plt.imshow( display_grid, aspect='auto', cmap='viridis' )