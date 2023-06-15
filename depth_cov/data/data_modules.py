import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

import depth_cov.data.data_loaders as data_loaders
from depth_cov.data.depth_transforms import TrainTransform, BaseTransform

class DepthCovDataModule(pl.LightningDataModule):

  def __init__(self, batch_size):
    super().__init__()

    self.batch_size = batch_size

    # TODO: Convert all to parameters
    self.img_size = torch.tensor([192, 256])
    
    self.train_transform = TrainTransform(self.img_size.tolist(), max_angle=15, crop_scale=True)
    self.val_transform = BaseTransform(self.img_size.tolist())

    self.num_workers = 8

  def setup(self, stage):

    self.train_dataset = data_loaders.ScanNetDataLoader(
        filename = './dataset/scannet_train.txt',
        transform = self.train_transform,
    )

    self.val_dataset = data_loaders.ScanNetDataLoader(
        filename = './dataset/scannet_val.txt',
        transform = self.val_transform,
    )

    # self.train_dataset = data_loaders.NyuDepthV2DataLoader(
    #     filename = './dataset/nyudepthv2_train.txt',
    #     transform = self.train_transform,
    # )

    # self.val_dataset = data_loaders.NyuDepthV2DataLoader(
    #     filename = './dataset/nyudepthv2_val.txt',
    #     transform = self.val_transform,
    # )

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=data_loaders.collate_fn)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=data_loaders.collate_fn)