from lib2to3.pytree import Base
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.transforms.functional as TF

from depth_cov.core.NonstationaryGpModule import NonstationaryGpModule
from depth_cov.data.depth_transforms import BaseTransform
from depth_cov.utils.utils import sample_coords, normalize_coordinates

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

# ScanNet loader
def get_rgb_depth(rgb_filename, depth_filename, size):
  rgb_pil = Image.open(rgb_filename)
  rgb = TF.to_tensor(rgb_pil)
  depth_np = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
  depth_np = depth_np.astype(np.float32) / 1000.0
  depth = torch.from_numpy(depth_np)
  depth = torch.unsqueeze(depth, dim=0)
  depth[depth<=0.0] = float('nan')
  log_depth = torch.log(depth)

  transform = BaseTransform(size)
  rgb, log_depth = transform(rgb, log_depth)

  return rgb, log_depth  

def get_data(num_samples, rgb_path, depth_path, size, device):
  rgb, gt_depth = get_rgb_depth(rgb_path, depth_path, size)
  rgb = rgb.unsqueeze(0)
  gt_depth = gt_depth.unsqueeze(0)

  coord_train, depth_train, batch_train = sample_coords(gt_depth, None, num_samples, mode="uniform")

  rgb = rgb.to(device)
  gt_depth = gt_depth.to(device)
  coord_train = coord_train.to(device)
  depth_train = depth_train.to(device)

  return rgb, gt_depth, coord_train, depth_train

def plot_rgb_and_coords(ax, img, sparse_coords):
  ax.clear()
  img_cpu = img.cpu()
  ax.imshow(torch.permute(img_cpu, (1,2,0)))
  sparse_coords_cpu = 2.5*sparse_coords.cpu()
  ax.scatter(x = sparse_coords_cpu[:,1], y = sparse_coords_cpu[:,0], c='r', s=30)

def plot_results(rgb, sparse_coords, sparse_depths, gt_depth, pred_depths, pred_vars, gaussian_covs):

  num_levels = len(gaussian_covs)
  fig, axs = plt.subplots(2, num_levels+1,figsize=(14,7))

  rgb_viz = torch.permute(rgb[0,...], (1,2,0)).cpu()
  depth_viz = gt_depth[0,0,...].cpu()

  sparse_coords_cpu = sparse_coords.cpu()
  sparse_depth_viz = torch.zeros_like(depth_viz)
  sparse_depth_viz[sparse_coords_cpu[0,:,0].long(), sparse_coords_cpu[0,:,1].long()] = sparse_depths[0,:,0].cpu()

  mask = ~gt_depth.isnan()
  vmin = torch.min(gt_depth[mask])
  vmax = torch.max(gt_depth[mask])

  axs[0,0].imshow(depth_viz, vmin=vmin, vmax=vmax, interpolation="nearest")
  axs[1,0].imshow(rgb_viz)
  axs[1,0].scatter(x = sparse_coords_cpu[0,:,1], y = sparse_coords_cpu[0,:,0], c='r', s=1)

  for i in range(num_levels):
    H = pred_depths[i].shape[-2]
    W = pred_depths[i].shape[-1]

    pred_depth_viz = pred_depths[i].cpu().detach()
    pred_depth_viz = pred_depth_viz[0,0,:,:]

    axs[0,i+1].imshow(pred_depth_viz, vmin=vmin, vmax=vmax)

    pred_vars_viz = pred_vars[i].cpu().detach()
    pred_vars_viz = pred_vars_viz[0,0,:,:]
    pred_std_viz = torch.sqrt(pred_vars_viz)

    axs[1,i+1].imshow(pred_std_viz, vmin=0.0, vmax=torch.max(pred_std_viz), cmap='jet')
  
  plt.show()

# TODO: Params in a config file?
def main(model_path, rgb_path, depth_path, num_samples, device):

  # Get data
  size = torch.Size([192, 256])
  rgb, gt_depth, coord_train, depth_train = get_data(num_samples, rgb_path, depth_path, size, device)
  mean_depth = torch.nanmean(depth_train)

  # Eval model
  model = NonstationaryGpModule.load_from_checkpoint(model_path, train_size=size)
  model.eval()
  model.to(device)
  # Run network
  gaussian_covs = model(rgb)
  # Condition on sparse inputs
  coords_train_norm = normalize_coordinates(coord_train, rgb.shape[-2:])
  pred_depths, pred_vars = model.condition(gaussian_covs, coords_train_norm, depth_train, mean_depth, size)

  # Plot results
  plot_results(rgb, coord_train, depth_train, gt_depth, pred_depths, pred_vars, gaussian_covs)

if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model_path = "models/scannet.ckpt"

  rgb_path = "dataset/examples/frame-000338.color.jpg"
  depth_path = "dataset/examples/frame-000338.depth.pgm"

  # Depth completion params
  num_samples = 128

  with torch.no_grad():
    main(model_path, rgb_path, depth_path, num_samples, device)