import os.path as osp

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, LidarSegPointCloud

from utils import uint16_MAX
from utils.geometry_utils import cart2sph

nusc = NuScenes(version='v1.0-mini', dataroot='/home/alec/data/sets/nuscenes', verbose=True)

range_image = np.zeros((360 * 2, 80))  # 0.5degree resolution in both axes
plt.ion()
figure, [ax1, ax2] = plt.subplots(2, 1, figsize=(10, 10))
ims = ax1.imshow(range_image.T, vmax=20)
bin = ax2.imshow(range_image.T, vmax=1)
plt.tight_layout()
plt.title("Frame 0")
plt.show()
print(len(nusc.sample))
for sample in nusc.sample:

    pointsensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
    pc_label_path = osp.join(nusc.dataroot, nusc.get('lidarseg', pointsensor['token'])['filename'])
    pc = LidarSegPointCloud(pcl_path, pc_label_path)
    if np.sum(pc.labels == nusc.lidarseg_name2idx_mapping['human.pedestrian.adult']) > 100:
        print(f"We shall save image {pointsensor['token']}")

        range_image = np.zeros((360 * 2, 80))  # 0.5degree resolution in both axes
        mask_image = range_image.copy()

        for p, l in zip(pc.points, pc.labels):
            rho, h_ang, v_ang = cart2sph(p[0], p[1], p[2])
            h_idx = int(np.rad2deg(h_ang * 2)) + 360
            v_idx = 10 - int(np.rad2deg(v_ang * 2))

            if v_idx >= 80 or v_idx < 0:
                continue
            range_image[h_idx % 720, v_idx] = rho
            mask_image[h_idx % 720, v_idx] = 1 if l == nusc.lidarseg_name2idx_mapping['human.pedestrian.adult'] and rho < 20 else 0

        ims.set_data(range_image.T)

        range_image = np.round(uint16_MAX / 40 * range_image)
        range_image[range_image > uint16_MAX] = uint16_MAX
        range_im = Image.fromarray(range_image.T.astype(np.uint16, casting="unsafe"))
        mask_im = Image.fromarray(mask_image.T.astype(np.uint8, casting="unsafe") * 255)
        #range_im.save(osp.join(out_folder, f"range/{j:05d}.png"))
        #mask_im.save(osp.join(out_folder, f"mask/{j:05d}.png"))

        bin.set_data(mask_image.T)
        figure.canvas.draw()
        figure.canvas.flush_events()

