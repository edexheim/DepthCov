from random import random
import torch
from torch.utils.data import Dataset

from depth_cov.utils.utils import get_coord_img

class DepthCompletionDataset(Dataset):
  def __init__(self, base_dataloader, num_samples):
    self.base_dataloader = base_dataloader
    self.num_samples = num_samples

  def __len__(self):
    return self.base_dataloader.__len__()

  def create_sparse_log_depth(self, rgb, log_depth):
    _, H, W = rgb.shape
    device = rgb.device

    log_depth_vec = torch.reshape(log_depth, (-1,))
    valid_depths = ~log_depth_vec.isnan() 
    weights = 1.0*valid_depths
    inds = torch.multinomial(weights, self.num_samples, replacement=False)

    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    coord_img = torch.dstack((y_coords, x_coords))
    coord_vec = torch.reshape(coord_img, (-1,2))
    sparse_coords = coord_vec[inds, :]
    sparse_log_depths = log_depth_vec[inds].unsqueeze(1)

    return sparse_coords, sparse_log_depths


  def __getitem__(self, idx):

    rgb, log_depth = self.base_dataloader.__getitem__(idx)

    sparse_coords, sparse_log_depth = self.create_sparse_log_depth(rgb, log_depth)
  
    mean_log_depth = torch.log(torch.median(torch.exp(log_depth[~log_depth.isnan()])))

    return rgb, log_depth, sparse_coords, sparse_log_depth, mean_log_depth