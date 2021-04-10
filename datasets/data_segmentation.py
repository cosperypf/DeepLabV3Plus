import os
import torch.utils.data as data
import numpy as np
import glob

from PIL import Image
import utils


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class DataSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    cmap = voc_cmap()

    def __init__(self,
                 image_set='train',
                 transform=None,
                 jpg_dir=None,
                 png_dir=None,
                 list_dir=None
                 ):
        self.transform = transform
        split_f = os.path.join(list_dir, image_set + ".txt")

        if not os.path.exists(split_f):
            print("check file exist:", split_f)
            raise ValueError('check file exist failed. ')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(jpg_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(png_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to GRAY image"""
        return mask

    @classmethod
    def decode_target_colorful(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]


class DataSegmentationTest(data.Dataset):
    def __init__(self, transform=None, test_dir=None, png_dir=None):
        self.transform = transform
        self.png_dir = png_dir

        # parse jpg and png
        self.images = glob.glob(os.path.join(test_dir, "*.jpg"))
        self.images.extend(glob.glob(os.path.join(test_dir, "*.png")))

        assert len(self.images) > 0, "No file type satisfied %s " % test_dir

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')

        if not utils.is_null_string(self.png_dir):
            mask_img = os.path.join(self.png_dir, os.path.basename(img_path))
            target = Image.open(mask_img)
        else:
            # if not exist mask image, use original picture instead
            target = Image.open(img_path)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to GRAY image"""
        return mask

    @classmethod
    def decode_target_colorful(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]
