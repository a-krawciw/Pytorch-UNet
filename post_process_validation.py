# Script which post-processes the output predictions of the validation.py script to generate an orientation float in the lidar frame
# by averaging the point wise class orientation of the people detected.

import pandas as pd # for processing csv
from math import atan2, ceil, pi
import os
import csv
import numpy as np
from PIL import Image


VALIDATION_PATH = "/home/jordy/aer1515/python_env2/Course Project/Pytorch-UNet/ped50_processed/combined_data/test/pred"
OUTPUT_FILE = "/home/jordy/aer1515/python_env2/Course Project/Pytorch-UNet/ped50_processed/combined_data/test/ped_orientation.csv"
n_classes = 9

# Open new csv
f = open(OUTPUT_FILE, 'w',encoding='UTF8', newline='')
writer = csv.writer(f)
header = ['index', 'orientation', 'class']
writer.writerow(header)

# Iterate through all files in the pred directory
i = 0
for file in sorted(os.listdir(VALIDATION_PATH)):
    filename= os.fsdecode(file)
    img = Image.open((VALIDATION_PATH + '/' + filename))
    numpy_img = np.asarray(img)

    # Iterate through all pixels
    orient_arr = []
    
    # Probably can vectorize this with numpy arrays to make it way faster, but for now its fine (approx 60-90 seconds for most runs)
    for x in numpy_img:
        for each_pixel in x:
            # Away from lidar is class 1, moving left is 2, moving towards is 3, moving right is 4
            pixel_class = (each_pixel * n_classes) / 255
            if pixel_class != 0:
                pixel_orient = (pixel_class-1) * ((2 * pi) / (n_classes-1))
                orient_arr.append(pixel_orient)
    
    # Average the orientation of all the pixels
    if len(orient_arr) != 0:
        avg_orient = sum(orient_arr) / len(orient_arr)
        # Write the result for this image to csv file
        row = [i,avg_orient, ceil(avg_orient / ((2*pi)/(n_classes-1)))]
        writer.writerow(row)
    else:
        row = [i,"NULL"]
        writer.writerow(row)

    i = i + 1

f.close()
print("success")