from time import time
import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import cv2

import pyrealsense2 as rs

import os
import re

from depth_cov.utils.lie_algebra import tq_to_pose
from depth_cov.odom.odom_geometry import resize_intrinsics, transformPoints
import depth_cov.data.depth_resize as depth_resize

# Assuming one by one loading
def odom_collate_fn(batch):
  assert len(batch) == 1
  return (batch[0][0], batch[0][1].unsqueeze(0))

class OdometryDataset(Dataset):
  def __init__(self, img_size):
    self.img_size = img_size

  def __len__(self):
    return self.data_len

  def __getitem__(self, idx):
    timestamp = self.load_timestamp(idx)
    rgb = self.load_rgb(idx)
    return timestamp, rgb


class ScanNetOdometryDataset(OdometryDataset):
  def __init__(self, seq_path, img_size, crop_size):
    super().__init__(img_size)

    self.seq_path = seq_path
    self.crop_size = crop_size

    rgb_list = []
    for file_name in os.listdir(seq_path):
      if file_name.endswith('.jpg'):
        file_stem = file_name.split('.')[0]
        rgb_list.append(os.path.join(seq_path, file_name))

    self.rgb_list = sorted(rgb_list)

    info_file = open(seq_path + "_info.txt")
    lines = info_file.readlines()
    
    color_width = self.line_to_np(lines[2])
    color_height = self.line_to_np(lines[3])
    size_orig = torch.tensor([color_height[0], color_width[0]])
    
    intrinsics_vec = self.line_to_np(lines[7])
    intrinsics_mat = np.reshape(intrinsics_vec, (4,4))
    intrinsics_mat = intrinsics_mat[:3,:3]
    intrinsics_orig = torch.from_numpy(intrinsics_mat)
    
    image_scale_factors = torch.tensor([480, 640])/size_orig # Images saved as this size
    self.intrinsics = resize_intrinsics(intrinsics_orig, image_scale_factors)
    self.intrinsics[0,2] -= self.crop_size
    self.intrinsics[1,2] -= self.crop_size
    image_scale_factors = torch.tensor(self.img_size)/torch.tensor([480-2*crop_size, 640-2*crop_size])
    self.intrinsics = resize_intrinsics(self.intrinsics, image_scale_factors) 

    self.data_len = len(self.rgb_list)

  def line_to_np(self, line):
    return np.fromstring(line.split(' = ')[1], sep=' ')

  def load_rgb(self, idx):
    bgr_np = cv2.imread(self.rgb_list[idx])
    rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
    rgb = TF.to_tensor(rgb_np)
    h, w = rgb.shape[-2:]
    rgb_crop = rgb[..., self.crop_size:(h-self.crop_size), self.crop_size:(w-self.crop_size)]
    rgb_r = TF.resize(rgb_crop, self.img_size, interpolation = TF.InterpolationMode.BILINEAR, antialias = True)
    return rgb_r

  # TODO: Is ScanNet always 30 FPS?
  def load_timestamp(self, idx):
    return idx/30.0

class TumOdometryDataset(OdometryDataset):
  def __init__(self, seq_path, img_size):
    super().__init__(img_size)

    self.seq_path = seq_path

    # rgb_ts rgb_filename depth_ts depth_filename
    rgb_file = open(seq_path + "rgb.txt")
    lines = rgb_file.readlines()
    self.ts_list = []
    self.rgb_list = []
    for i in range(3, len(lines)): # Skip info from first 3 lines
      line_list = lines[i].split()
      self.ts_list.append(float(line_list[0]))
      self.rgb_list.append(os.path.join(seq_path, line_list[1]))
    
    self.data_len = len(self.rgb_list)

    # dataset_ind = int(re.findall(r'\d+', seq_path)[-1])
    match = re.search('freiburg(\d+)', seq_path)
    dataset_ind = int(match.group(1))
    self.setup_camera_vars(dataset_ind)

  def setup_camera_vars(self, dataset_ind):
    size_orig = torch.tensor([480, 640])
    image_scale_factors = torch.tensor(self.img_size)/size_orig

    ## ROS Default 
    # self.intrinsics_orig = torch.tensor([ [ 525.0,    0.0,  319.5], 
    #                                       [   0.0,  525.0,  239.5],
    #                                       [   0.0,    0.0,    1.0]] )
    # self.distorton = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    if dataset_ind == 1:
      intrinsics_orig = torch.tensor([ [ 517.3,    0.0,  318.6], 
                                            [   0.0,  516.5,  255.3],
                                            [   0.0,    0.0,    1.0]] )
      distortion = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])
    elif dataset_ind == 2:
      intrinsics_orig = torch.tensor([ [ 520.9,    0.0,  325.1], 
                                            [   0.0,  521.0,  249.7],
                                            [   0.0,    0.0,    1.0]] )
      distortion = np.array([0.2312, -0.7849, -0.0033, -0.0001, 0.9172])
    elif dataset_ind == 3:
      intrinsics_orig = torch.tensor([ [ 535.4,    0.0,  320.1 ], 
                                            [   0.0,  539.2,  247.6 ],
                                            [   0.0,    0.0,    1.0]] )
      distortion = None
    else:
      raise ValueError("TumOdometryDataset with dataset ind " + dataset_ind + " is not a valid dataset.")

    ## NOTE: With 0 distortion, getOptimalNewCameraMatrix gives different K, 
    # and initUndistortRectifyMap will have a map with values at the borders... 

    # Setup distortion
    if distortion is not None:
      orig_img_size = [size_orig[1].item(), size_orig[0].item()]
      K = intrinsics_orig.numpy()
      # alpha = 0.0 means invalid pixels are cropped, while 1.0 means all original pixels are present in new image
      K_u, validPixROI = cv2.getOptimalNewCameraMatrix(K, distortion, orig_img_size, alpha=0, newImgSize=orig_img_size)
      # TODO: What type to use for maps?
      self.map1, self.map2 = cv2.initUndistortRectifyMap(K, distortion, None, K_u, orig_img_size, cv2.CV_32FC1)
      intrinsics_orig = torch.from_numpy(K_u)
    else:
      self.map1, self.map2 = None, None
      intrinsics_orig = intrinsics_orig

    self.intrinsics = resize_intrinsics(intrinsics_orig, image_scale_factors)

  def load_rgb(self, idx):
    bgr_np = cv2.imread(self.rgb_list[idx])
    rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)

    # Undistort/resize
    # Use precalculated initUndistortRectifyMap for faster dataloading
    if self.map1 is not None:
      rgb_np_u = cv2.remap(rgb_np, self.map1, self.map2, cv2.INTER_LINEAR)
    else:
      rgb_np_u = rgb_np

    new_img_size = [self.img_size[1], self.img_size[0]]
    rgb_np_resized = cv2.resize(rgb_np_u, new_img_size, interpolation = cv2.INTER_LINEAR)

    rgb = TF.to_tensor(rgb_np_resized)
    return rgb

  def load_depth(self, idx):
    depth_np = cv2.imread(self.depth_list[idx], cv2.IMREAD_ANYDEPTH)
    depth_np = depth_np.astype(np.float32) / 5000.0
    depth = torch.from_numpy(depth_np)
    depth = torch.unsqueeze(depth, dim=0)
    depth[depth<=0.0] = float('nan')
    return depth

  def load_pose(self, idx):
    return self.poses[idx:idx+1,:,:]

  def load_timestamp(self, idx):
    return self.ts_list[idx]

class RealsenseDataset(IterableDataset):
  def __init__(self, img_size, cfg):
    super().__init__()
    self.img_size = img_size
    self.cfg = cfg

    self.start()

  def start(self):

    config = rs.config()
    config.enable_stream(
        stream_type=rs.stream.color, 
        width=self.cfg["width"],
        height=self.cfg["height"],
        framerate=self.cfg["fps"])

    self.pipeline = rs.pipeline()
    profile = self.pipeline.start(config)

    rgb_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    rgb_intrinsics = rgb_profile.get_intrinsics()

    size_orig = torch.tensor([rgb_intrinsics.height, rgb_intrinsics.width])
    image_scale_factors = torch.tensor(self.img_size)/size_orig

    intrinsics_orig = torch.tensor([  [ rgb_intrinsics.fx,    0.0,  rgb_intrinsics.ppx], 
                                      [   0.0,  rgb_intrinsics.fy,  rgb_intrinsics.ppy],
                                      [   0.0,    0.0,    1.0]] )
    distortion = np.asarray(rgb_intrinsics.coeffs)

    ## NOTE: With 0 distortion, getOptimalNewCameraMatrix gives different K, 
    # and initUndistortRectifyMap will have a map with values at the borders... 

    # Setup distortion
    if distortion is not None:
      orig_img_size = [size_orig[1].item(), size_orig[0].item()]
      K = intrinsics_orig.numpy()
      # alpha = 0.0 means invalid pixels are cropped, while 1.0 means all original pixels are present in new image
      K_u, validPixROI = cv2.getOptimalNewCameraMatrix(K, distortion, orig_img_size, alpha=0, newImgSize=orig_img_size)
      # TODO: What type to use for maps?
      self.map1, self.map2 = cv2.initUndistortRectifyMap(K, distortion, None, K_u, orig_img_size, cv2.CV_32FC1)
      intrinsics_orig = torch.from_numpy(K_u)
    else:
      self.map1, self.map2 = None, None
      intrinsics_orig = intrinsics_orig

    self.intrinsics = resize_intrinsics(intrinsics_orig, image_scale_factors)

  def shutdown(self):
    self.pipeline.stop()

  def __len__(self):
    return 1.0e+10

  def __iter__(self):
    return self

  def __next__(self):

    frameset = self.pipeline.wait_for_frames()
    
    timestamp = frameset.get_timestamp()
    timestamp /= 1000.0 # original in ms

    rgb_frame = frameset.get_color_frame()
    rgb_np = np.asanyarray(rgb_frame.get_data())

    # Undistort
    if self.map1 is not None:
      rgb_np_u = cv2.remap(rgb_np, self.map1, self.map2, cv2.INTER_LINEAR)
    else:
      rgb_np_u = rgb_np
    new_img_size = [self.img_size[1], self.img_size[0]]
    rgb_np_resized = cv2.resize(rgb_np_u, new_img_size, interpolation = cv2.INTER_LINEAR)
    rgb = TF.to_tensor(rgb_np_resized)

    return timestamp, rgb