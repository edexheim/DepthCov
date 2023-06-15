import torch
import open3d as o3d
import open3d.visualization.gui as gui

from depth_cov.odom.OdomWindow import OdomWindow
from depth_cov.odom.odom_datasets import TumOdometryDataset, ScanNetOdometryDataset

import yaml

def main():
  torch.manual_seed(0)

  img_size = [192, 256]  

  # dataset_dir = "/path_to/scannet/scene0155_00/"
  # dataset = ScanNetOdometryDataset(dataset_dir, img_size, crop_size=10)

  dataset_dir = "/path_to/tum/rgbd_dataset_freiburg3_long_office_household/"
  # dataset_dir = "/path_to/tum/rgbd_dataset_freiburg2_desk/"
  dataset = TumOdometryDataset(dataset_dir, img_size)

  ## Parameters
  with open('./config/open3d_viz.yml', 'r') as file:
    viz_cfg = yaml.safe_load(file)
  # Odometry setup
  with open('./config/tum.yml', 'r') as file:
    slam_cfg = yaml.safe_load(file)

  # Open3D visualization setup
  app = gui.Application.instance
  app.initialize()
  is_live = False # If processing faster than data itself, wait to match data FPS
  viz_window = OdomWindow(is_live, viz_cfg, slam_cfg, dataset)
  app.run()

if __name__ == "__main__":
  main()