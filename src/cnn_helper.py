"""
CNN helper functions for loading data
"""

import PIL
import matplotlib.pyplot as plt


def crop_one_img(img_path, out_dir='', car='unspecified'):
    """
    Crop an image to contain mainly car features

    Args:
        img_path: input path
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