import argparse
import logging
import os
from os.path import splitext
from pathlib import Path

import numpy
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

from utils import uint16_MAX
from utils.data_loading import BasicDataset
from unet import UNet
from utils.geometry_utils import sph2cart
from utils.utils import plot_img_and_mask


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(BasicDataset, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])
        
        if net.n_classes > 2:
            full_mask = probs.squeeze(0)
            full_mask = probs.squeeze().cpu()
        else:
            full_mask = tf(probs.cpu()).squeeze() # Doesnt seem to work for multi classes

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))
def image2sph(im):
    #np.rad2deg(az * 2) + 360, 20 - np.rad2deg(el * 2)
    iter = np.nditer(im, flags=['multi_index'], order='C')
    for rho in iter:
        idx_x = iter.multi_index[0]
        idx_y = iter.multi_index[1]
        az = np.deg2rad(idx_y - 360) / 2.0
        el = np.deg2rad(20 - idx_x) / 2.0
        yield float(rho) / uint16_MAX * 40, az, el

def euclidean_3d(points, query):
    return np.sqrt((points['x']-query['x'])**2 + (points['y'] - query['y'])**2 + (points['z'] - query['z'])**2)

def filter_by_radius(point_cloud, radius, dist_func=None):
    if dist_func is None:
        dist_func = euclidean_3d

    mask_shape = len(point_cloud)
    output_mask = np.full(mask_shape, False)
    for i, point in enumerate(point_cloud):
        mask = np.full(mask_shape, True)
        mask[i] = False
        dists = dist_func(point_cloud[mask], point)
        output_mask[i] = np.min(dists) > radius

    return np.delete(point_cloud, output_mask)

PLOT = False

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    dataset_dir = Path("Ped50Data/_2019-02-09-13-04-06")
    images_dir = dataset_dir / "range"
    out_mask_dir = dataset_dir / "pred"
    masks_dir = dataset_dir / "mask"
    model_dir = Path(".")
    #images_dir = Path("./data/imgs")
    #out_mask_dir = Path("./val/preds")
    #masks_dir = Path("./data/masks")

    out_mask_dir.mkdir(exist_ok=True)

    range_files = [os.path.join(images_dir, file) for file in sorted(os.listdir(images_dir)) if not file.startswith('.')]
    mask_files = [os.path.join(masks_dir, file) for file in sorted(os.listdir(masks_dir)) if not file.startswith('.')]
    out_files = [os.path.join(out_mask_dir, file) for file in sorted(os.listdir(images_dir)) if not file.startswith('.')]

    net = UNet(n_channels=1, n_classes=9, bilinear=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model ')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(model_dir / "model.pth", map_location=device))

    logging.info('Model loaded!')
    if model_dir == dataset_dir:
        print("Training errors")

    ious = []
    pres = []
    recalls = []
    for i, (range_file, mask_file) in enumerate(zip(range_files, mask_files)):
        #logging.info(f'\nPredicting image {range_file} ...')
        img = Image.open(range_file)
        true_mask = Image.open(mask_file)
        true_mask = np.array(true_mask) / 255
        true_mask = true_mask.astype(bool)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=1,
                           out_threshold=0.5,
                           device=device)

        out_filename = out_files[i]
        result = mask_to_image(mask)
        result.save(out_filename)
        #logging.info(f'Mask saved to {out_filename}')

        range_image = np.array(img)
        mask = mask[-1]
        mask = mask.astype(bool)
        valid_idxs = range_image > 0
        tp_mask = numpy.bitwise_and(mask, true_mask)
        tn_mask = numpy.bitwise_and(np.bitwise_not(mask), np.bitwise_not(true_mask))
        fp_mask = numpy.bitwise_and(mask, np.bitwise_not(true_mask))
        fn_mask = numpy.bitwise_and(np.bitwise_not(mask), true_mask)
        matches = true_mask[valid_idxs] == mask[valid_idxs]
        not_matches = true_mask[valid_idxs] != mask[valid_idxs]
        tp = np.sum(tp_mask[valid_idxs])
        fp = np.sum(fp_mask[valid_idxs])
        tn = np.sum(tn_mask[valid_idxs])
        fn = np.sum(fn_mask[valid_idxs])

        out_vis = 255*np.ones((80, 720, 3), dtype=np.uint8)
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)
        grey = (127, 127, 127)
        out_vis[tp_mask] = green
        out_vis[fp_mask] = red
        out_vis[fn_mask] = blue
        out_vis[np.bitwise_not(valid_idxs)] = grey

        if tp > 0:
            iou = tp/(tp+fp+fn)
            ious.append(iou)
            precision = tp/(tp+fp)
            pres.append(precision)
            recall = tp/(tp+fn)
            recalls.append(recall)
            print(f"IOU={iou}, Precision={precision}, Recall={recall}")

            true_mask_flat = np.array(true_mask).flatten()
            pred_mask_flat = np.array(mask).flatten()

            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('true_class', bool), ('pred_class', bool)]
            point_cloud = []
            #for i, (rho, az, el) in enumerate(image2sph(range_image)):
            #    if rho > 0:
            #        x, y, z = sph2cart(rho, az, el)
            #        point_cloud.append((x, y, z, true_mask_flat[i] != 0, pred_mask_flat[i] != 0))
            #pc_arr = np.array(point_cloud, dtype=dtype)
            #pc_arr = filter_by_radius(pc_arr, 0.05)



        else:
            print("Blank")



        if PLOT:
            logging.info(f'Visualizing results for image {range_file}, close to continue...')
            plt.figure(figsize=(8, 5))
            plt.imshow(out_vis)
            plt.title("TP: Green, FP: red, FN: blue")
            plt.show()
            #plot_img_and_mask(true_mask, mask)
    print(f"Mean IOU={np.mean(ious)}")
    print(f"Mean Precision={np.mean(pres)}")
    print(f"Mean Recall={np.mean(recalls)}")
    plt.hist(ious)
    plt.show()