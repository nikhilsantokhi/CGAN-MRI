# Import necessary packages
import pytorch_lightning as pl
import os
from torch.utils.data import DataLoader
from data import CineDataset


class OCMRDataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset_dir: str, # path to dataset
            acceleration: str, # choose acceleration (i.e. R=2, R=4, R=8)
            batch_size: int = 8,
            num_workers: int = 4,
            train_transform=None,
            val_transform=None,
            test_transform=None,
    ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.acceleration = acceleration
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    # create the batch dataloaders for train/val/test
    def _create_data_loader(self, dataset, shuffle: bool):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    def train_dataloader(self):
        al_dir = os.path.join(self.dataset_dir, 'AL', self.acceleration, 'train')
        gt_dir = os.path.join(self.dataset_dir, 'GT', 'train')

        dataset = CineDataset(al_dir, gt_dir, transform=self.train_transform, do_augment=True)

        return self._create_data_loader(dataset, shuffle=True)

    def val_dataloader(self):
        al_dir = os.path.join(self.dataset_dir, 'AL', self.acceleration, 'val')
        gt_dir = os.path.join(self.dataset_dir, 'GT', 'val')

        dataset = CineDataset(al_dir, gt_dir, transform=self.val_transform, do_augment=False)

        return self._create_data_loader(dataset, shuffle=False)

    def test_dataloader(self):
        al_dir = os.path.join(self.dataset_dir, 'AL', self.acceleration, 'test')
        gt_dir = os.path.join(self.dataset_dir, 'GT', 'test')

        dataset = CineDataset(al_dir, gt_dir, transform=self.test_transform, do_augment=False)

        return self._create_data_loader(dataset, shuffle=False)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('OCMR Data')

        parser.add_argument('--dataset-dir', type=str, required=True)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--num_workers', type=int, default=4)

        return parser