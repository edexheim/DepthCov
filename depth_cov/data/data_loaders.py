import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

import h5py
import cv2

import depth_cov.data.depth_resize as depth_resize

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class BaseDataset(Dataset):
  def __init__(self, filename, transform):
    self.filename = filename
    self.transform = transform

    with open(filename) as f:
      self.lines = f.read().split('\n')
    
    self.data_len = len(self.lines)

  def __len__(self):
    return self.data_len

  def normalize_depth(self, depth, valid_mask):
    nonzero_ind_tuple = torch.nonzero(valid_mask, as_tuple=True)
    log_depth = torch.empty_like(depth)
    log_depth[:] = float('nan')
    log_depth[nonzero_ind_tuple] = torch.log(depth[nonzero_ind_tuple])
    return log_depth, nonzero_ind_tuple

  def __getitem__(self, idx):

    rgb, depth = self.get_rgb_depth(idx)
    depth[depth<=0.0] = float('nan')

    # Transform
    if self.transform is not None:
      rgb, depth = self.transform(rgb, depth)

    # Prepare log depth
    log_depth = torch.log(depth)

    return rgb, log_depth

class NyuDepthV2DataLoader(BaseDataset):
  def __init__(self, filename, transform):
    super().__init__(filename, transform)

  def get_rgb_depth(self, idx):
    line = self.lines[idx]
    h5f = h5py.File(line, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = TF.to_tensor(rgb)
    depth = torch.from_numpy(np.array(h5f['depth']))
    depth = torch.unsqueeze(depth, dim=0)
    
    # NYU Specific preprocessing
    rgb = TF.resize(rgb, [240, 320], interpolation = TF.InterpolationMode.BILINEAR, antialias = True)
    rgb = TF.center_crop(rgb, [228, 304])
    depth = depth_resize.resize_depth(depth, mode="nearest_neighbor", size=[240, 320])
    depth = TF.center_crop(depth, [228, 304])

    return rgb, depth

class ScanNetDataLoader(BaseDataset):
  def __init__(self, filename, transform):
    super().__init__(filename, transform)

  def load_depth(self, filename):
    depth_np = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
    depth_np = depth_np.astype(np.float32) / 1000.0
    return depth_np

  def get_filenames(self, idx):
    line = self.lines[idx]
    rgb_filename, depth_filename = line.split(',', 1)
    return rgb_filename, depth_filename

  def get_rgb_depth(self, idx):
    rgb_filename, depth_filename = self.get_filenames(idx)
    rgb_pil = Image.open(rgb_filename)
    rgb = TF.to_tensor(rgb_pil)
    depth_np = self.load_depth(depth_filename)
    depth = torch.from_numpy(depth_np)
    depth = torch.unsqueeze(depth, dim=0)
    return rgb, depth  

class TumDataLoader(BaseDataset):
  def __init__(self, filename, transform):
    super().__init__(filename, transform)

  def load_depth(self, filename):
    depth_np = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
    depth_np = depth_np.astype(np.float32) / 5000.0
    return depth_np

  def get_filenames(self, idx):
    line = self.lines[idx]
    rgb_ts, rgb_filename, depth_ts, depth_filename = line.split()
    return rgb_filename, depth_filename

  def get_rgb_depth(self, idx):
    rgb_filename, depth_filename = self.get_filenames(idx)
    
    dir = self.filename.rsplit("/", 1)[0] + "/"
    rgb_pil = Image.open(dir+rgb_filename)
    rgb = TF.to_tensor(rgb_pil)

    depth_np = self.load_depth(dir+depth_filename)
    depth = torch.from_numpy(depth_np)
    depth = torch.unsqueeze(depth, dim=0)

    return rgb, depth  