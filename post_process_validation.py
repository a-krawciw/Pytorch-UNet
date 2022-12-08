# Script which post-processes the output predictions of the validation.py script to generate an orientation float in the lidar frame
# by averaging the point wise class orientation of the people detected.

import pandas as pd # for processing csv
from math import atan2, ceil, pi
import os
import csv
import numpy as np
from PIL import Image


VALIDATION_PATH = "Ped50Data/test/pred"
OUTPUT_FILE = "Ped50Data/test/ped_orientation.csv"
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

    if np.mean(numpy_img > 0) != 0:
        p_class = np.round(numpy_img / 255.0 * n_classes)
        avg_orient = np.mean(p_class[p_class>0], axis=None) - 1
        # Write the result for this image to csv file
        row = [i, avg_orient*np.pi/4, np.round(avg_orient)]
        writer.writerow(row)
    else:
        row = [i,"NULL"]
        writer.writerow(row)


    i = i + 1

f.close()
print("success")