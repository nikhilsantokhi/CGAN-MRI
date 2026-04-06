# Import necessary packages
import os
import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop, RandomRotation

class CineDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            al_dir,
            gt_dir,
            transform=None,
            do_augment=False
    ):
        self.al_dir = al_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.do_augment = do_augment

        self.al_images = sorted(os.listdir(al_dir))

    def __len__(self):
        return len(self.al_images) # returns the length of a batch

    def __getitem__(self, idx):
        # grabs the input and ground truth images
        al_fname = self.al_images[idx]
        gt_fname = al_fname.replace('_AL_', '_GT_')

        al_path = os.path.join(self.al_dir, al_fname)
        gt_path = os.path.join(self.gt_dir, gt_fname)

        if not os.path.exists(gt_path):
            raise FileNotFoundError

        al_img = Image.open(al_path)
        gt_img = Image.open(gt_path)

        # apply augmentations
        if self.do_augment:
            al_img, gt_img = self.augment(al_img, gt_img)

        if self.transform:
            al_img = self.transform(al_img)
            gt_img = self.transform(gt_img)

        return al_img, gt_img # returns the images of a batch

    def augment(self, al, gt):
        # horizontal flip
        if torch.rand(1) < 0.5:
            al = F.hflip(al)
            gt = F.hflip(gt)

        # vertical flip
        if torch.rand(1) < 0.5:
            al = F.vflip(al)
            gt = F.vflip(gt)

        # random rotation
        angle = RandomRotation.get_params([-10, 10])
        al = F.rotate(al, angle)
        gt = F.rotate(gt, angle)

        # random resized crop
        i, j, h, w = RandomCrop.get_params(al, output_size=(256, 256))
        al = F.crop(al, i, j, h, w)
        gt = F.crop(gt, i, j, h, w)

        return al, gt