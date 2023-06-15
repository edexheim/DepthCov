
import torch
import torchvision.transforms.functional as TF
import torch.multiprocessing as mp

import time

from depth_cov.core.NonstationaryGpModule import NonstationaryGpModule
from depth_cov.core.samplers import sample_sparse_coords_norm
from depth_cov.odom.photo_tracking import photo_tracking_pyr, precalc_jacobians
from depth_cov.odom.photo_utils import setup_test_coords
from depth_cov.utils.image_processing import ImageGradientModule, ImagePyramidModule, IntrinsicsPyramidModule, DepthPyramidModule
from depth_cov.utils.lie_algebra import invertSE3
from depth_cov.utils.utils import init_gpu, str_to_dtype, swap_coords_xy, normalize_coordinates, get_test_coords, fill_image
from depth_cov.odom.multiprocessing import release_data
from depth_cov.odom.odom_geometry import backprojection, get_T_w_curr, get_rel_pose, get_aff_w_curr, get_rel_aff

# Do not declare any CUDA tensors in init function
class DepthOdometry(mp.Process):
  def __init__(self, cfg, intrinsics, img_size, waitev):
    super().__init__()

    self.cfg = cfg
    self.device = cfg["device"]
    self.dtype = str_to_dtype(cfg["dtype"])

    self.intrinsics = intrinsics
    self.img_size = img_size

    self.waitev = waitev
    self.mapping_init = False

  def init_basic_vars(self):
    self.intrinsics = self.intrinsics.to(device=self.device, dtype=self.dtype)

    start_level = self.cfg["pyr"]["start_level"]
    end_level = self.cfg["pyr"]["end_level"]
    depth_interp_mode = self.cfg["pyr"]["depth_interp_mode"]

    intrinsics_pyr_module = IntrinsicsPyramidModule(start_level, end_level, self.device)
    self.intrinsics_pyr = intrinsics_pyr_module(self.intrinsics, [1.0, 1.0])

    c = 1
    self.gradient_module = ImageGradientModule(channels=c, device=self.device, dtype=self.dtype)
    self.img_pyr_module = ImagePyramidModule(c, start_level, end_level, self.device, dtype=self.dtype)
    self.depth_pyr_module = DepthPyramidModule(start_level, end_level, depth_interp_mode, self.device)

  def reset_one_way_vars(self):
    self.last_one_way_num_pixels = self.img_size[-1] * self.img_size[-2] /  \
        (self.cfg["grad_pruning"]["nonmax_suppression_window"] * self.cfg["grad_pruning"]["nonmax_suppression_window"])

  def run(self):
    init_gpu(self.device)
    self.init_basic_vars()
    self.init_kf_vars()
    self.reset_one_way_vars()
    self.T_w_rec_last = None
    
    while True:
      # Check if new keyframe reference
      kf_data = self.kf_ref_queue.pop_until_latest(block=False, timeout=0.01)
      if kf_data is not None:
        if kf_data[0] == "end":
          self.tracking_pose_queue.push(("end",))
          break
        else:
          ref_start = time.time()
          self.update_kf_reference(kf_data)
          ref_end = time.time()
          # print("Update ref time: ", 1.0e3*(ref_end - ref_start), "ms")
          # print("ODOM UPDATED KEYFRAME")
      release_data(kf_data)

      # Get new RGB
      data = self.rgb_queue.pop(timeout=0.01)
      if data is not None:
        if data[0] == "end":
          # Signal mapping queue
          self.frame_queue.push(("end",))
        elif not self.mapping_init:
          timestamp, rgb = data
          self.frame_queue.push(("init", timestamp, rgb.clone()))
        else:
          self.handle_frame(data)
      release_data(data)

    self.waitev.wait()

    return

  def handle_frame(self, data):
    timestamp, rgb = data
    
    # track_start = time.time()

    # Track against reference
    img_pyr = self.prep_tracking_img(rgb)
    self.T_curr_kf, self.aff_curr_kf, coords_curr, depth_curr = photo_tracking_pyr( \
        self.T_curr_kf, self.aff_curr_kf,
        self.vals_pyr, self.P_pyr, self.dI_dT_pyr, self.intrinsics_pyr,
        img_pyr, self.cfg["sigmas"]["photo"], self.cfg["term_criteria"])
    
    coords_curr_norm = normalize_coordinates(coords_curr, img_pyr[-1].shape[-2:])

    # Send tracked pose
    T_w_curr = self.get_curr_world_pose()
    self.tracking_pose_queue.push((timestamp, T_w_curr.clone()))

    # track_end = time.time()
    # print("Track time: ", 1.0e3*(track_end - track_start), "ms")

    # Decide if keyframe
    depth_kf = self.P_pyr[-1][...,2]
    reproj_depth = fill_image(coords_curr, depth_curr, img_pyr[-1].shape[-2:])
    new_kf = self.check_keyframe(depth_kf, torch.count_nonzero(~torch.isnan(reproj_depth)), self.T_curr_kf)
    if new_kf:
      kf_data = ("keyframe", rgb.clone(), self.T_curr_kf, self.aff_curr_kf, self.kf_received_ts, coords_curr_norm, depth_curr, timestamp)
      self.frame_queue.push(kf_data)
      self.last_kf_sent_ts = timestamp
      kf_end = time.time()
    else:
      # Try to see if add one way frame
      new_one_way_frame = self.check_one_way_frame(depth_kf, torch.count_nonzero(~torch.isnan(reproj_depth)), self.T_curr_kf, T_w_curr)
      if new_one_way_frame:
        gx, gy = self.gradient_module(img_pyr[-1])
        img_and_grads = torch.cat((img_pyr[-1], gx, gy), dim=1)

        one_way_data = ("one-way", rgb.clone(), self.T_curr_kf, self.aff_curr_kf, self.kf_received_ts, timestamp)
        self.frame_queue.push(one_way_data)
        self.last_rec_sent_ts = timestamp
        one_way_end = time.time()
  
  def update_kf_reference(self, kf_data):
    timestamp, img, kf_pose, kf_aff, depth = kf_data

    # Update curr frame to kf variables
    if timestamp > self.kf_received_ts and self.mapping_init:
      self.T_w_f = get_T_w_curr(self.T_w_kf, self.T_curr_kf)
      self.T_curr_kf = get_rel_pose(self.T_w_f, kf_pose)

      self.aff_w_f = get_aff_w_curr(self.aff_w_kf, self.aff_curr_kf)
      self.aff_curr_kf  = get_rel_aff(self.aff_w_f, kf_aff)
      
      # Don't have this info but assume full image
      self.reset_one_way_vars()

    elif not self.mapping_init:
      self.mapping_init = True
      self.last_kf_sent_ts = timestamp

    # If same image reference, only update depth vars
    if timestamp != self.kf_received_ts:
      img_pyr = self.img_pyr_module(img)

      self.coords_pyr = []
      self.vals_pyr = []
      self.img_grads_pyr = []
      for i in range(len(img_pyr)):
        gx, gy = self.gradient_module(img_pyr[i])

        test_coords = get_test_coords(img_pyr[i].shape[-2:], device=self.device)

        # Prune coords
        if img_pyr[i].shape[-2] >= self.cfg["grad_pruning"]["start_res"][0] or \
            img_pyr[i].shape[-1] >= self.cfg["grad_pruning"]["start_res"][1]:
          mean_sq_grad_norm = torch.sqrt(torch.square(gx) + torch.square(gy))
          max_mean_sq_grad_norm, max_indices = torch.nn.functional.max_pool2d(mean_sq_grad_norm, 
              kernel_size=self.cfg["grad_pruning"]["nonmax_suppression_window"], return_indices=True)
          coords_mask = max_mean_sq_grad_norm > self.cfg["grad_pruning"]["grad_norm_thresh"]
          indices = max_indices[coords_mask]
          test_coords = test_coords[:,indices,:]
        
        self.coords_pyr.append(test_coords)

        vals = img_pyr[i][:,:,test_coords[0,:,0],test_coords[0,:,1]]
        self.vals_pyr.append(vals)

        gx = gx[0,:,test_coords[0,:,0],test_coords[0,:,1]]
        gy = gy[0,:,test_coords[0,:,0],test_coords[0,:,1]]
        dI_dw = torch.stack((gx, gy), dim=2)
        dI_dw = torch.permute(dI_dw, (1,0,2))
        self.img_grads_pyr.append(dI_dw)

    # Compute variables involving geometry regardless
    self.P_pyr = []
    self.dI_dT_pyr = []
    depth_pyr = self.depth_pyr_module(depth)
    for i in range(len(depth_pyr)):
      test_coords = self.coords_pyr[i]
      depths = depth_pyr[i][:,:,test_coords[0,:,0],test_coords[0,:,1]]
      depths = torch.permute(depths, (0,2,1))

      test_coords_xy = swap_coords_xy(test_coords)
      P, _ = backprojection(self.intrinsics_pyr[i], test_coords_xy, depths)
      dI_dT = precalc_jacobians(self.img_grads_pyr[i], P, self.vals_pyr[i], self.intrinsics_pyr[i])
      
      self.P_pyr.append(P)
      self.dI_dT_pyr.append(dI_dT)

    self.kf_received_ts = timestamp
    self.T_w_kf = kf_pose
    self.aff_w_kf = kf_aff

  def get_curr_world_pose(self):
    T_w_curr = get_T_w_curr(self.T_w_kf, self.T_curr_kf)
    return T_w_curr

  def get_curr_world_aff(self):
    aff_curr = get_aff_w_curr(self.aff_w_kf, self.aff_curr_kf)
    return aff_curr

  def prep_tracking_img(self, rgb):
    img_tracking = TF.rgb_to_grayscale(rgb)

    c, h, w = img_tracking.shape[-3:]
    device = img_tracking.device
    img_pyr = self.img_pyr_module(img_tracking)

    return img_pyr

  def get_img_gradients(self, img_pyr):
    c = img_pyr[-1].shape[-3]
    device = img_pyr[-1].device

    img_and_grads = []
    for l in range(len(img_pyr)):
      gx, gy = self.gradient_module(img_pyr[l])
      img_and_grads_level = torch.cat((img_pyr[l], gx, gy), dim=1)
      img_and_grads.append(img_and_grads_level)
    
    return img_and_grads

  # These are variables relative to a reference keyframe
  # Affine brightness parameters are not global for that frame!
  def init_kf_vars(self):
    self.T_curr_kf = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0)
    self.aff_curr_kf = torch.zeros((1,2,1), device=self.device, dtype=self.dtype)
    self.last_one_way_num_pixels = self.img_size[-1] * self.img_size[-2]

    self.last_kf_sent_ts = torch.zeros(1, device=self.device, dtype=self.dtype)
    self.kf_received_ts = torch.zeros(1, device=self.device, dtype=self.dtype)

  def check_keyframe(self, depth_kf, num_reproj_depth, T_curr_kf):
    new_kf = False

    num_kf_pixels = self.vals_pyr[-1].shape[2]

    # Need to have received new kf from mapping to avoid immediately setting keyframe
    if self.last_kf_sent_ts <= self.kf_received_ts:
      median_depth = torch.median(depth_kf)
      kf_dist = torch.linalg.norm(T_curr_kf[:,:3,3])
      # print("KF frame dist: ", kf_dist, median_depth)
      if kf_dist > self.cfg["keyframing"]["kf_depth_motion_ratio"] * median_depth:
        new_kf = True
      elif self.cfg["keyframing"]["kf_num_pixels_frac"] > num_reproj_depth/num_kf_pixels:
        new_kf = True
    else:
      print("Keyframe ", self.last_kf_sent_ts, " still not received, continue tracking against kf ", self.kf_received_ts)

    return new_kf 

  def check_one_way_frame(self, depths_kf, num_reproj_depth, T_curr_kf, T_w_curr):
    new_one_way_frame = False

    # Closest distance to kf or one way frame
    kf_dist = torch.linalg.norm(T_curr_kf[:,:3,3])
    if self.T_w_rec_last is not None:
      last_one_way_dist = torch.linalg.norm(T_w_curr[:,:3,3] - self.T_w_rec_last[:,:3,3])
      frame_dist = min(kf_dist, last_one_way_dist)
    else:
      frame_dist = kf_dist

    median_depth = torch.median(depths_kf)
    if frame_dist > self.cfg["keyframing"]["one_way_depth_motion_ratio"] * median_depth:
      new_one_way_frame = True
    elif self.cfg["keyframing"]["one_way_pixel_frac"] > num_reproj_depth/self.last_one_way_num_pixels:
      new_one_way_frame = True

    if new_one_way_frame:
      self.last_one_way_num_pixels = num_reproj_depth
      self.T_w_rec_last = T_w_curr

    return new_one_way_frame