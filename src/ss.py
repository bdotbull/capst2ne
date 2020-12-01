"""
Program to rapidly take screenshots for gathering data
"""

from PIL import ImageGrab
from datetime import datetime
from time import sleep

max_ss = 2000    # number of Screenshots to take
ss_count = 0     # counter for loop
car = "oct"      # three-letter signifier for which car is being recorded

# Initial wait to allow for alt+tab into game
sleep(5)

while ss_count < max_ss:
    im = ImageGrab.grab()
    dt = datetime.now()

    file_path_prefix = r"..\\data\\img\\screenshots\\grey_env\\oct\\"   # make sure to change filepath to match `car`
    fname = "{}{}_{}.jpg".format(file_path_prefix, car, ss_count)
    im.save(fname, 'jpeg')

    print(f'Saved {car} # {ss_count}')

    ss_count += 1