import torch
import open3d as o3d
import open3d.visualization.gui as gui

from depth_cov.odom.OdomWindow import OdomWindow
from depth_cov.odom.odom_datasets import RealsenseDataset

import yaml

def main():
  torch.manual_seed(0)

  img_size = [192, 256]  
  with open('./config/realsense.yml', 'r') as file:
    rs_cfg = yaml.safe_load(file)

  dataset = RealsenseDataset(img_size, rs_cfg)

  ## Parameters
  with open('./config/open3d_viz.yml', 'r') as file:
    viz_cfg = yaml.safe_load(file)
  # Odometry setup
  with open('./config/visual_odom.yml', 'r') as file:
    slam_cfg = yaml.safe_load(file)

  # Open3D visualization setup
  app = gui.Application.instance
  app.initialize()
  is_live = True # Ensures no throttling of data loop to match FPS, just grabs image as quickly as possible
  viz_window = OdomWindow(is_live, viz_cfg, slam_cfg, dataset)
  app.run()

if __name__ == "__main__":
  main()