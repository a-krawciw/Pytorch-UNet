# Python script to extract the per pixel orientation from the ped50 dataset ped_motion.csv files

# Input: ped_motion.csv (file with pedestrian relative position in lidar frame)
# Output: ped_orientation.csv (file with ground truth per pixel orientation of the person in the ped50 run in lidar frame

import pandas as pd # for processing csv
from math import atan2, ceil, pi
import os
import csv


# Takes in yaw value from [-pi pi] and the number of orientation classes and outputs which class corresponds to that yaw value
def classify_orientation(yaw, n_classes):
    if yaw < 0:
        yaw = yaw + (2*pi)
    rad_interval = (2*pi)/(n_classes-1)
    yaw_class = ceil(yaw / rad_interval)

    # Away from lidar is class 1, moving left is 2, moving towards is 3, moving right is 4

    return yaw_class

run_list = [
            "_2019-02-09-13-04-06", # 5m  SEQ: 1 
            #"_2019-02-09-13-04-51", # 5m  SEQ: 2 
            #"_2019-02-09-13-07-14", # 10m SEQ: 3 
            #"_2019-02-09-13-08-12", # 10m SEQ: 4 # Note this one appears malformed, dont use
            #"_2019-02-09-15-16-08", # 5m  SEQ: 15 # I think the mask is empty on this one
            #"_2019-02-09-15-16-50", # 5m  SEQ: 16 # has a bad mask
            #"_2019-02-09-15-18-22", # 10m SEQ: 17 # has bad mask
            #"_2019-02-09-15-19-03", # 10m SEQ: 18 # has bad mask
            #"_2019-02-09-15-52-16", # 5m  SEQ: 28
            #"_2019-02-09-15-55-03", # 10m SEQ: 29
            #"_2019-02-09-14-56-32",  # Toward SEQ: 38
            #"_2019-02-09-15-32-23", # Toward SEQ: 40
            #"_2019-02-09-14-59-13", # Curved SEQ: 46 # Quite far
            #"_2019-02-09-15-00-09", # Curved SEQ: 47 # Quite far
            #"_2019-02-09-15-01-27", # ZigZag SEQ: 48 # Quite far
            #"_2019-02-09-15-34-30", # Curved SEQ: 49
            #"_2019-02-09-15-36-46" # ZigZag SEQ: 51
]

PED_RUN_DIR = "/home/jordy/aer1515/python_env2/Course Project/Pytorch-UNet/ped50_processed/_2019-02-09-15-36-46"
PED_MOTION_PATH = PED_RUN_DIR + "/ped_motion.csv"
PED_MASK_PATH = PED_RUN_DIR + "/mask"
OUTPUT_FILE = PED_RUN_DIR + "/mask/ped_orientation.csv"
LOOKAHEAD = 30
n_classes = 9

# Check the length of the mask directory, this controls our loop intervals because the ped_motion.csv has more data then we need
iterations=len([entry for entry in os.listdir(PED_MASK_PATH ) if os.path.isfile(os.path.join(PED_MASK_PATH , entry))]) - 1

data = pd.read_csv(PED_MOTION_PATH)
px = pd.DataFrame(data, columns=["smooth_px_v"])
py = pd.DataFrame(data, columns=["smooth_py_v"])
orientation_arr = []

# Open new csv
f = open(OUTPUT_FILE, 'w',encoding='UTF8', newline='')
writer = csv.writer(f)
header = ['index', 'orientation', 'class']
writer.writerow(header)

for i in range(iterations+1):
    if i + LOOKAHEAD < iterations:
        x1 = px.values[i][0]
        x2 = px.values[i+LOOKAHEAD][0]
        dx = x2-x1
        y1 = py.values[i][0]
        y2 = py.values[i+LOOKAHEAD][0]
        dy = y2-y1
        yaw = atan2(dy,dx)
        orientation_arr.append(yaw)

        # Write row to csv
        yaw_class = classify_orientation(yaw,n_classes)
        row = [i,yaw, yaw_class]
        writer.writerow(row)


    # If we have run out of horizon, take the last valid orientation to not have changed
    else:
        orientation_arr.append(orientation_arr[-1])

        # Write row to csv
        yaw_class = classify_orientation(yaw,n_classes)
        row = [i,yaw, yaw_class]
        writer.writerow(row)

f.close()
print("success")
