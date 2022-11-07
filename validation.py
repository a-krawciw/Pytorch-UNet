import argparse
import logging
import os
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
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

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


PLOT = True

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    #images_dir = Path("/home/alec/Documents/UofT/AER1515/coat_pass2/range")
    #out_mask_dir = Path("/home/alec/Documents/UofT/AER1515/coat_pass2/preds")
    #masks_dir = Path("/home/alec/Documents/UofT/AER1515/coat_pass2/mask")
    images_dir = Path("./data/imgs")
    out_mask_dir = Path("./val/preds")
    masks_dir = Path("./data/masks")

    range_files = [os.path.join(images_dir, file) for file in sorted(os.listdir(images_dir)) if not file.startswith('.')]
    mask_files = [os.path.join(masks_dir, file) for file in sorted(os.listdir(masks_dir)) if not file.startswith('.')]
    out_files = [os.path.join(out_mask_dir, file) for file in sorted(os.listdir(images_dir)) if not file.startswith('.')]

    net = UNet(n_channels=1, n_classes=2, bilinear=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model ')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load("./checkpoints/checkpoint_epoch10.pth", map_location=device))

    logging.info('Model loaded!')

    for i, (range_file, mask_file) in enumerate(zip(range_files, mask_files)):
        logging.info(f'\nPredicting image {range_file} ...')
        img = Image.open(range_file)
        true_mask = Image.open(mask_file)
        true_mask = np.array(true_mask)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=1,
                           out_threshold=0.5,
                           device=device)

        out_filename = out_files[i]
        result = mask_to_image(mask)
        result.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')

        range_image = np.array(img)
        mask = mask[-1]*255
        valid_idxs = range_image > 0
        matches = true_mask[valid_idxs] == mask[valid_idxs]
        not_matches = true_mask[valid_idxs] != mask[valid_idxs]
        tp = float(np.sum(true_mask[valid_idxs][matches] > 0))
        fp = float(np.sum(true_mask[valid_idxs][not_matches] == 0))
        tn = float(np.sum(true_mask[valid_idxs][matches] == 0))
        fn = float(np.sum(true_mask[valid_idxs][not_matches] > 0))
        if tp > 0:
            iou = tp/(tp+fp+fn)
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            print(f"IOU={iou}, Precision={precision}, Recall={recall}")
        else:
            print("Blank")

        if PLOT:
            logging.info(f'Visualizing results for image {range_file}, close to continue...')
            plot_img_and_mask(true_mask, mask)
