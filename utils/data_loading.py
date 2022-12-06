import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

class BasicDataset(Dataset):
    def __init__(self, images_dir: str or Path, masks_dir: str or Path, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess(self, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.array(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))
        else:
            img_ndarray = (img_ndarray / 255)

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def load_files(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(
            mask_file) == 1, f'Either no mask or multiple masks found for the ID {name + self.mask_suffix + ".*"}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        return img, mask


    def __getitem__(self, idx):
        img, mask = self.load_files(idx)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class Ped50Dataset(BasicDataset):

    def __init__(self, ped50_root_dir: Path):
        self.root_dir = ped50_root_dir
        super().__init__(ped50_root_dir / "range", ped50_root_dir / "mask")
        self.orientation_file = self.root_dir / "mask/ped_orientation.csv"
        orient_data = pd.read_csv(self.orientation_file)
        self.orient_class = pd.DataFrame(orient_data, columns=["class"])

    def load_files(self, idx):
        img, mask = super().load_files(idx)
        orient_class = self.orient_class.values[idx][0]
        mask *= orient_class
        return img, mask


class ShuffledDataset(BasicDataset):
    def __init__(self, dataset_to_shuffle: BasicDataset):
        super().__init__(dataset_to_shuffle.images_dir, dataset_to_shuffle.masks_dir, dataset_to_shuffle.scale)
        self.dataset = dataset_to_shuffle
        self.offset = 0


    def load_files(self, idx):
        img, mask = self.dataset.load_files(idx)
        idx_offset = np.random.randint(1, img.shape[2])
        self.offset = idx_offset
        img_cop = img.copy()
        mask_cop = mask.copy()

        mask[:, 0:-idx_offset] = mask_cop[:, idx_offset:]
        mask[:, -idx_offset:] = mask_cop[:, 0:idx_offset]
        img[:, :, 0:-idx_offset] = img_cop[:, :, idx_offset:]
        img[:, :, -idx_offset:] = img_cop[:, :, 0:idx_offset]

        return img, mask

class ShuffledOrientationDataset(ShuffledDataset):
    def load_files(self, idx):
        img, mask = super().load_files(idx)
        binned_offset = np.round(self.offset / 720 * 8)
        mask[mask > 0] = np.mod(mask[mask > 0] - 1 + binned_offset, 8) + 1
        return img, mask

class JitteredDataset(BasicDataset):
    def __init__(self, dataset_to_shuffle: BasicDataset, jitter_max = 5):
        super().__init__(dataset_to_shuffle.images_dir, dataset_to_shuffle.masks_dir, dataset_to_shuffle.scale)
        self.dataset = dataset_to_shuffle
        self.jitter = jitter_max


    def load_files(self, idx):
        img, mask = self.dataset.load_files(idx)
        idx_offset = np.random.randint(-self.jitter, self.jitter)
        img_out = np.zeros(img.shape)
        mask_out = np.zeros(mask.shape)


        if idx_offset >= 0:
            img_region = img[:, idx_offset:, :]
            mask_region = mask[idx_offset:, :]
            mask_out[:mask_region.shape[0], :] = mask_region
            img_out[:, :img_region.shape[1], :] = img_region
        else:
            img_region = img[:, :idx_offset, :]
            mask_region = mask[:idx_offset, :]
            mask_out[-mask_region.shape[0]:, :] = mask_region
            img_out[:, -img_region.shape[1]:, :] = img_region

        return img, mask