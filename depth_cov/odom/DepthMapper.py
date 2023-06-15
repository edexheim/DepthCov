import time

import torch
import torchvision.transforms.functional as TF
import torch.multiprocessing as mp

from depth_cov.core.NonstationaryGpModule import NonstationaryGpModule
from depth_cov.core.samplers import sample_sparse_coords_norm
from depth_cov.odom.depth_prior_factors import (
  linearize_sparse_depth_prior, construct_sparse_depth_prior_system_batch, construct_sparse_depth_prior_system
)
from depth_cov.odom.odom_geometry import predict_log_depth_img, backprojection, predict_log_depth_img_no_J, resize_intrinsics, get_T_w_curr, get_aff_w_curr
from depth_cov.odom.odom_opt_utils import get_depth_inds, get_dims, solve_system, update_vars
from depth_cov.odom.photo_mapping import create_photo_system
from depth_cov.odom.photo_utils import setup_test_coords
from depth_cov.odom.pose_prior_factors import linearize_pose_prior
from depth_cov.odom.scalar_prior_factors import linearize_scalar_prior, linearize_multi_scalar_prior
from depth_cov.utils.lie_algebra import normalizeSE3_inplace
from depth_cov.utils.utils import init_gpu, str_to_dtype, get_coord_img, normalize_coordinates, get_test_coords, swap_coords_xy, unnormalize_coordinates
from depth_cov.odom.multiprocessing import release_data
from depth_cov.odom.TwoFrameSfm import TwoFrameSfm
from depth_cov.utils.image_processing import ImageGradientModule

# Do not declare any CUDA tensors in init function
class DepthMapper(mp.Process):
  def __init__(self, cfg, intrinsics, waitev):
    super().__init__()

    self.cfg = cfg
    self.device = cfg["device"]
    self.dtype = str_to_dtype(cfg["dtype"])

    self.intrinsics = intrinsics

    self.waitev = waitev
    self.is_init = False
  
  def init_basic_vars(self):
    self.intrinsics = self.intrinsics.to(device=self.device, dtype=self.dtype)

    c = 1
    self.gradient_module = ImageGradientModule(channels=c, device=self.device, dtype=self.dtype)

  def init_keyframe_vars(self):
    # Bookkeeping
    self.timestamps = []
    # Inputs
    self.rgb = torch.empty((0), device=self.device, dtype=self.dtype) # For visualization
    self.img_and_grads = torch.empty((0), device=self.device, dtype=self.dtype)
    self.stdev_inv = torch.empty((0), device=self.device, dtype=self.dtype)
    # Sliding window unknown variables
    self.poses = torch.empty((0), device=self.device, dtype=self.dtype)
    self.mean_log_depths = torch.empty((0), device=self.device, dtype=self.dtype)
    self.aff_params = torch.empty((0), device=self.device, dtype=self.dtype)
    # Sparse depth vars - note that tensors are padded with zeros if varying num points
    self.sparse_coords_norm_list = [] # For visualization and bookkeeping num points
    self.sparse_log_depth = torch.empty((0), device=self.device, dtype=self.dtype)
    self.Knm_Kmminv = torch.empty((0), device=self.device, dtype=self.dtype)
    self.dr1_ds = torch.empty((0), device=self.device, dtype=self.dtype)
    self.dr1_dd = torch.empty((0), device=self.device, dtype=self.dtype)
    self.H1_s_s = torch.empty((0), device=self.device, dtype=self.dtype)
    self.H1_s_d = torch.empty((0), device=self.device, dtype=self.dtype)
    self.H1_d_d = torch.empty((0), device=self.device, dtype=self.dtype)

    self.pose_history = torch.empty((0), device=self.device, dtype=self.dtype)
    
  def init_prior_vals(self):
    self.window_full = False
    self.pose_anchor = torch.empty((0), device=self.device, dtype=self.dtype)
    self.scale_anchor = torch.empty((0), device=self.device, dtype=self.dtype)
    self.aff_anchor = torch.empty((0), device=self.device, dtype=self.dtype)
    self.sparse_log_depth_anchor = torch.empty((0), device=self.device, dtype=self.dtype)

  def run(self):
    init_gpu(self.device)
    self.init_basic_vars()
    self.load_model()
    self.init_keyframe_vars()
    self.init_prior_vals()
    self.clear_one_way_frames()
    self.reset_iteration_vars(converged=True)
    self.two_frame_sfm = TwoFrameSfm(self.cfg, self.intrinsics, self.model, self.cov_level, self.network_size)

    while True:
      kf_updated = False
      if not self.is_init:
        data = self.frame_queue.pop_until_latest(block=True, timeout=0.01)
        if data is not None and data[0] == "init":
          timestamp, rgb = data[1:]
          self.is_init, img_and_grads_curr, T_curr_kf, aff_curr_kf, sparse_log_depth_kf, coords_curr, depth_curr \
              = self.two_frame_sfm.handle_frame(rgb)
          if self.is_init:
            # Initialize reference keyframe
            self.init_keyframe(rgb, self.two_frame_sfm.gaussian_covs, self.two_frame_sfm.sparse_coords_norm,
                self.two_frame_sfm.img_and_grads[-1], self.two_frame_sfm.pose_init, sparse_log_depth_kf, self.two_frame_sfm.aff_init, self.two_frame_sfm.mean_log_depth_init, timestamp)
            self.two_frame_sfm.delete_init_reference()
            # Initialize second keyframe
            pose_curr = get_T_w_curr(self.two_frame_sfm.pose_init, T_curr_kf)
            aff_curr = get_aff_w_curr(self.two_frame_sfm.aff_init, aff_curr_kf)
            coords_curr_norm = normalize_coordinates(coords_curr, rgb.shape[-2:]) 
            self.add_keyframe(rgb, pose_curr, aff_curr, coords_curr_norm, depth_curr, timestamp)
            kf_updated = True
        release_data(data)
      else:
        # Handle one frame at a time
        data = self.frame_queue.pop(timeout=0.01)
        if data is not None:
          if data[0] == "one-way":
            # print("MAPPING NEW 1-WAY")
            # Do something with one-way frame
            rgb, pose_curr_kf, aff_curr_kf, kf_timestamp, timestamp = data[1:]
            kf_ind = self.find_kf_from_timestamp(kf_timestamp)
            pose_w_init = self.get_curr_world_pose(pose_curr_kf, kf_ind)
            aff_w_init = self.get_curr_world_aff(aff_curr_kf, kf_ind)
            self.add_one_way_frame(rgb, pose_w_init, aff_w_init, timestamp)
          elif data[0] == "keyframe":
            # print("MAPPING NEW KF")
            # Send keyframe before new one added
            kf_viz_data = self.get_kf_viz_data()
            self.kf_viz_queue.push(kf_viz_data)
            # Do something with kf data
            rgb, pose_curr_kf, aff_curr_kf, kf_timestamp, reproj_coords_norm, reproj_depths, timestamp = data[1:]
            kf_ind = self.find_kf_from_timestamp(kf_timestamp)
            pose_w_init = self.get_curr_world_pose(pose_curr_kf, kf_ind)
            aff_w_init = self.get_curr_world_aff(aff_curr_kf, kf_ind)
            self.add_keyframe(rgb, pose_w_init, aff_w_init, reproj_coords_norm, reproj_depths, timestamp)
            kf_updated = True
          elif data[0] == "end":
            break
        release_data(data)
      
      if self.is_init and not self.converged:
        # while not self.converged:
        # torch.cuda.synchronize()
        # start = time.time()
        self.converged = self.iterate()
        # torch.cuda.synchronize()
        # end = time.time()
        # print("iterate time: ", end-start)
        kf_updated = True
        # TODO: Should we update this every time?

      if kf_updated:
        # Send updated keyframe data to queue
        kf_ref_data = self.get_kf_ref_data()
        self.kf_ref_queue.push(kf_ref_data)

    self.kf_ref_queue.push(("end",))
    self.kf_viz_queue.push(("end",))

    self.waitev.wait()

    return
  
  # Same as add_keyframe but given init depth variables, so don't calculate
  def init_keyframe(self, rgb, gaussian_covs, sparse_coords_norm, img_and_grads, \
      pose_init, sparse_log_depth_init, aff_init, mean_log_depth_init, timestamp):
    
    with torch.no_grad():
      L_mm, Knm_Kmminv, stdev_inv = self.prep_predictor(gaussian_covs, sparse_coords_norm)

    self.initialize_unknown_vars(pose_init, mean_log_depth_init, aff_init)  
    self.initialize_photo_vars(rgb, img_and_grads, stdev_inv)
    self.initialize_sparse_depth_vars(sparse_coords_norm, sparse_log_depth_init, Knm_Kmminv, L_mm)

    self.timestamps = [timestamp]

  def add_keyframe(self, rgb, pose_init, aff_init, curr_coords_norm, curr_depths, timestamp):   

    img_and_grads = self.get_img_and_grads(rgb)

    gaussian_covs = self.run_model(rgb)
    sparse_coords_norm = self.sample_coords(gaussian_covs)
    L_mm, Knm_Kmminv, stdev_inv = self.prep_predictor(gaussian_covs, sparse_coords_norm)
    sparse_log_depth_init, mean_log_depth_init = self.initialize_log_depth(L_mm, Knm_Kmminv, stdev_inv, curr_coords_norm, curr_depths)

    self.initialize_unknown_vars(pose_init, mean_log_depth_init, aff_init)  
    self.initialize_photo_vars(rgb, img_and_grads, stdev_inv)
    self.initialize_sparse_depth_vars(sparse_coords_norm, sparse_log_depth_init, Knm_Kmminv, L_mm)

    self.window_cat_helper_list(self.timestamps, timestamp, self.get_kf_start_window_ind())

    self.reset_iteration_vars()

  def clear_one_way_frames(self):
    # Bookkeeping
    self.recent_timestamps = []
    # Inputs
    self.recent_img_and_grads = torch.empty((0), device=self.device, dtype=self.dtype)
    # Variables (no depths)
    self.recent_poses = torch.empty((0), device=self.device, dtype=self.dtype)
    self.recent_aff_params = torch.empty((0), device=self.device, dtype=self.dtype)

  def get_img_and_grads(self, rgb):
    gray = TF.rgb_to_grayscale(rgb)
    gx, gy = self.gradient_module(gray)
    img_and_grads = torch.cat((gray, gx, gy), dim=1)
    return img_and_grads

  def find_kf_from_timestamp(self, kf_timestamp):
    kf_ind = None
    for i in range(len(self.timestamps)-1,-1,-1):
      if kf_timestamp == self.timestamps[i]:
        kf_ind = i
        break
    return kf_ind
  
  def get_curr_world_pose(self, pose_curr_kf, kf_ind):
    T_w_curr = get_T_w_curr(self.poses[kf_ind:kf_ind+1,...], pose_curr_kf)
    return T_w_curr
  
  def get_curr_world_aff(self, aff_curr_kf, kf_ind):
    aff_curr = get_aff_w_curr(self.aff_params[kf_ind:kf_ind+1,...], aff_curr_kf)
    return aff_curr

  def add_one_way_frame(self, rgb, pose_init, aff_init, timestamp):
    img_and_grads = self.get_img_and_grads(rgb)

    recent_ind = self.get_recent_start_window_ind()

    self.window_cat_helper_list(self.recent_timestamps, timestamp, recent_ind)

    self.window_cat_helper_tensor(self.recent_img_and_grads, img_and_grads, recent_ind)
    self.window_cat_helper_tensor(self.recent_poses, pose_init, recent_ind)
    self.window_cat_helper_tensor(self.recent_aff_params, aff_init, recent_ind)

    self.reset_iteration_vars()

  def load_model(self):
    self.cov_level = -1
    self.network_size = torch.tensor([192, 256], device=self.device) 
    self.network_size_list = self.network_size.tolist()

    self.model = NonstationaryGpModule.load_from_checkpoint(self.cfg["model_path"], train_size=self.network_size)
    self.model.eval()
    self.model.to(self.device)
    self.model.to(torch.float)

  def run_model(self, rgb):
    # Gaussian covs
    rgb_r = TF.resize(rgb, self.network_size_list, interpolation=TF.InterpolationMode.BILINEAR, antialias=True).float()
    with torch.no_grad():
      gaussian_covs = self.model(rgb_r)
      # TODO: Should we convert gaussian covs to double? Or handle float most places?
      for l in range(len(gaussian_covs)):
        gaussian_covs[l] = gaussian_covs[l].to(dtype=self.dtype)
    
    return gaussian_covs

  def sample_coords(self, gaussian_covs):
    with torch.no_grad():
      # Select sparse coords
      sparse_coords_norm = sample_sparse_coords_norm(gaussian_covs, 
        self.cfg["sampling"]["max_samples"], mode=self.cfg["sampling"]["mode"], max_stdev_thresh=self.cfg["sampling"]["max_stdev_thresh"],
        model=self.model, model_level=self.cov_level,
        mask = None)
      
    return sparse_coords_norm

  def prep_predictor(self, gaussian_covs, sparse_coords_norm):
    b, _, h, w = gaussian_covs[-1].shape
    img_size = (h,w)
    device = gaussian_covs[-1].device

    test_coord_img = get_coord_img(img_size, device=device)
    test_coords_norm = normalize_coordinates(test_coord_img.to(self.dtype), img_size)
    test_coords_norm = torch.reshape(test_coords_norm, (b, h*w, 2))

    with torch.no_grad():
      L_mm, E_m = self.model.get_covariance_chol(gaussian_covs, self.cov_level, sparse_coords_norm)
      K_mn = self.model.get_cross_covariance(gaussian_covs, self.cov_level, sparse_coords_norm, test_coords_norm, E1=E_m)
      K_nn_diag = self.model.get_diagonal_covariance(gaussian_covs, self.cov_level, test_coords_norm)
    
    Kmminv_Kmn = torch.cholesky_solve(K_mn, L_mm, upper=False)
    Knm_Kmminv = torch.transpose(Kmminv_Kmn, dim0=-2, dim1=-1)
    Knm_Kmminv = torch.reshape(Knm_Kmminv, (b, h, w, -1))

    with torch.no_grad():
      log_depth_var = K_nn_diag - torch.sum(K_mn * Kmminv_Kmn, dim=1) + self.model.get_var(self.cov_level)
    log_depth_var = torch.reshape(log_depth_var, (b, 1, h, w))
    stdev_inv = 1.0/torch.sqrt(log_depth_var)

    return L_mm, Knm_Kmminv, stdev_inv

  def initialize_log_depth(self, L_mm, Knm_Kmminv, stdev_inv, curr_coords_norm, curr_depths):
    mean_log_depth_init = torch.log(torch.median(curr_depths, dim=1, keepdim=True).values)
    
    with torch.no_grad():
      log_depth_obs = torch.log(curr_depths)
      x_samples = swap_coords_xy(curr_coords_norm.unsqueeze(1))
      interp_mode="bilinear"
      Knm_Kmminv_curr = torch.nn.functional.grid_sample(torch.permute(Knm_Kmminv, (0,3,1,2)), x_samples,
          mode=interp_mode, padding_mode='zeros', align_corners=False)
      Knm_Kmminv_curr = torch.permute(Knm_Kmminv_curr[:,:,0,:], (0,2,1))
      stdev_inv_curr = torch.nn.functional.grid_sample(stdev_inv, x_samples,
          mode=interp_mode, padding_mode='zeros', align_corners=False)
      stdev_inv_curr = torch.permute(stdev_inv_curr[:,:,0,:], (0,2,1))
      sparse_log_depth, _ = self.model.solve_compact_depth_hierarchical(L_mm, Knm_Kmminv_curr, mean_log_depth_init, log_depth_obs, stdev_inv_curr)

    return sparse_log_depth, mean_log_depth_init

  def window_cat_helper_tensor(self, var, new_var, i):
    var.set_(torch.cat((var[i:,...], new_var), dim=0))
  
  def window_cat_helper_list(self, var, new_var, i):
    del var[:i]
    var.append(new_var)

  def window_cat_helper_tensor_padded(self, var, new_var, new_var_pad, i):
    padded_new_var = torch.nn.functional.pad(new_var, pad=new_var_pad, mode='constant', value=0)
    var.set_(torch.cat((var[i:,...], padded_new_var), dim=0))

  # Keeps all but first keyframe once capacity is reached
  def get_kf_start_window_ind(self):
    num_max_frames = self.cfg["graph"]["num_keyframes"]
    ind = -num_max_frames+1
    return ind

  def get_recent_start_window_ind(self):
    num_max_frames = self.cfg["graph"]["num_one_way_frames"]
    ind = -num_max_frames+1
    return ind

  # Remove oldest keyframe and reset anchors
  def initialize_unknown_vars(self, pose_init, mean_log_depth_init, aff_init):
    self.window_full = self.poses.shape[0] >= self.cfg["graph"]["num_keyframes"]

    if self.window_full:
      self.pose_history = torch.cat((self.pose_history, self.poses[0:1,...]), dim=0)

    # Add new frame and remove oldest frame if window is full
    kf_start_ind = self.get_kf_start_window_ind()
    normalizeSE3_inplace(pose_init)
    self.window_cat_helper_tensor(self.poses, pose_init, kf_start_ind)
    self.window_cat_helper_tensor(self.mean_log_depths, mean_log_depth_init, kf_start_ind)
    self.window_cat_helper_tensor(self.aff_params, aff_init, kf_start_ind)

    # Set anchors
    self.pose_anchor = self.poses[0:1,...].clone()
    self.scale_anchor = self.mean_log_depths[0:1,...].clone()
    self.aff_anchor = self.aff_params[0:1,...].clone()

  def initialize_photo_vars(self, rgb, img_and_grads, stdev_inv):    
    kf_start_ind = self.get_kf_start_window_ind()
    self.window_cat_helper_tensor(self.rgb, rgb, kf_start_ind)
    self.window_cat_helper_tensor(self.img_and_grads, img_and_grads, kf_start_ind)
    self.window_cat_helper_tensor(self.stdev_inv, stdev_inv, kf_start_ind)

  def initialize_sparse_depth_vars(self, sparse_coords_norm, sparse_log_depth_init, Knm_Kmminv, L_mm):
    kf_start_ind = self.get_kf_start_window_ind()

    dr_ds, dr_dd, H_s_s, H_s_d, H_d_d = linearize_sparse_depth_prior(L_mm)

    new_depth_dim = sparse_coords_norm.shape[1]
    new_var_pad =  self.cfg["sampling"]["max_samples"] - new_depth_dim

    self.window_cat_helper_tensor_padded(self.sparse_log_depth, sparse_log_depth_init, (0,0,0,new_var_pad,0,0), kf_start_ind)
    self.window_cat_helper_tensor_padded(self.Knm_Kmminv, Knm_Kmminv, (0,new_var_pad), kf_start_ind)
    self.window_cat_helper_tensor_padded(self.dr1_ds, dr_ds, (0,0,0,0,0,new_var_pad,0,0), kf_start_ind)
    self.window_cat_helper_tensor_padded(self.dr1_dd, dr_dd, (0,new_var_pad,0,0,0,new_var_pad,0,0), kf_start_ind)
    self.window_cat_helper_tensor(self.H1_s_s, H_s_s, kf_start_ind)
    self.window_cat_helper_tensor_padded(self.H1_s_d, H_s_d, (0,new_var_pad,0,0,0,0), kf_start_ind)
    self.window_cat_helper_tensor_padded(self.H1_d_d, H_d_d, (0,new_var_pad,0,new_var_pad,0,0), kf_start_ind)

    self.window_cat_helper_list(self.sparse_coords_norm_list, sparse_coords_norm, kf_start_ind)

    # Set anchor
    self.sparse_log_depth_anchor = self.sparse_log_depth[0:1,:,:].clone()
  
  def reset_iteration_vars(self, converged=False):
    self.converged = converged
    self.iter = 0
    self.total_err_prev = float("inf")

  def get_safe_ind(self, ind):
    if ind == -1 or ind >= self.poses.shape[0]:
      ind = self.poses.shape[0]-1
    return ind

  def get_depth_imgs(self):
    log_depth_imgs = predict_log_depth_img_no_J(self.sparse_log_depth, self.mean_log_depths, self.Knm_Kmminv)
    depth_imgs = torch.exp(log_depth_imgs[:,:,:,0]).unsqueeze(1)
    return depth_imgs

  def get_kf_ref_data(self, ind=-1):
    ind = self.get_safe_ind(ind)

    timestamp = self.timestamps[ind]
    img = self.img_and_grads[ind:ind+1,0:1,:,:]
    pose = self.poses[ind:ind+1,...]
    aff = self.aff_params[ind:ind+1,...]

    log_depth, _, _ = predict_log_depth_img(
        self.sparse_log_depth[ind:ind+1,...], 
        self.mean_log_depths[ind:ind+1,...], 
        self.Knm_Kmminv[ind:ind+1,...])
    depth_img = torch.exp(log_depth[0:1,:,:,0]).unsqueeze(1)

    return timestamp, img, pose, aff, depth_img

  def get_kf_viz_data(self, ind=-1):
    ind = self.get_safe_ind(ind)

    timestamps = self.timestamps
    rgb = self.rgb[ind:ind+1,...]
    poses = self.poses.clone()

    log_depth, _, _ = predict_log_depth_img(
        self.sparse_log_depth[ind:ind+1,...], 
        self.mean_log_depths[ind:ind+1,...], 
        self.Knm_Kmminv[ind:ind+1,...])
    depth_img = torch.exp(log_depth[0:1,:,:,0]).unsqueeze(1)
    
    sparse_coords_norm = self.sparse_coords_norm_list[ind]

    rec_poses = self.recent_poses.clone()
    kf_pairs = self.kf_pairs
    one_way_pairs = self.one_way_pairs

    return timestamps, rgb, poses, depth_img, sparse_coords_norm, rec_poses, kf_pairs, one_way_pairs

  def get_depth_img(self, ind=-1):
    ind = self.get_safe_ind(ind)

    log_depth, _, _ = predict_log_depth_img(
        self.sparse_log_depth[ind:ind+1,...], 
        self.mean_log_depths[ind:ind+1,...], 
        self.Knm_Kmminv[ind:ind+1,...])

    depth_img = torch.exp(log_depth[0:1,:,:,0])
    return depth_img


  def iterate(self):

    num_keyframes = self.poses.shape[0]
    num_recent_frames = self.recent_poses.shape[0]
    depth_inds = get_depth_inds(self.sparse_coords_norm_list)

    D, D_kf_T, D_kf_aff, D_rec_T, D_rec_aff, D_kf_s, D_kf_d = get_dims(num_keyframes, num_recent_frames, depth_inds)

    H = torch.zeros((D,D), device=self.device, dtype=self.dtype)
    g = torch.zeros((D), device=self.device, dtype=self.dtype)

    # Photometric error
    # torch.cuda.synchronize()
    # start = time.time()
    mean_sq_photo_err, self.kf_pairs, self.one_way_pairs= create_photo_system(
        self.poses, self.mean_log_depths, self.sparse_log_depth, self.aff_params,
          self.recent_poses, self.recent_aff_params, self.recent_img_and_grads,
          self.img_and_grads, self.Knm_Kmminv, depth_inds, self.intrinsics, 
          self.cfg["sigmas"], self.cfg["photo_construction"], self.cfg["graph"],
          H, g)
    # torch.cuda.synchronize()
    # end = time.time()
    # print("photo time: ", end-start)

    # Depth priors for each keyframe
    depth_prior_err1 = construct_sparse_depth_prior_system_batch(self.sparse_log_depth, self.mean_log_depths,       
        H, g, D_kf_s, D_kf_d, self.dr1_ds, self.dr1_dd, self.H1_s_s, self.H1_s_d, self.H1_d_d)

    # Gauge freedom priors
    pose_prior_err = linearize_pose_prior(self.poses[0:1,...], self.pose_anchor[0:1,...], 
        H, g, D_kf_T[0], sigma=self.cfg["sigmas"]["pose_prior"])

    scale_prior_err = linearize_scalar_prior(self.mean_log_depths[0,...], self.scale_anchor[0,...], H, g, D_kf_s[0], sigma=self.cfg["sigmas"]["scale_prior"])
    
    aff_scale_inds = [D_kf_aff[0][0], D_kf_aff[0][0]+1]
    aff_scale_err = linearize_scalar_prior(self.aff_params[0,0:1,:], self.aff_anchor[0,0:1,:], H, g, aff_scale_inds, sigma=self.cfg["sigmas"]["scale_prior"])
    aff_bias_inds = [D_kf_aff[0][0]+1, D_kf_aff[0][0]+2]
    aff_bias_err = linearize_scalar_prior(self.aff_params[0,1:2,:], self.aff_anchor[0,1:2,:], H, g, aff_bias_inds, sigma=self.cfg["sigmas"]["scale_prior"])

    # TODO: Marginalization instead?
    first_depth_prior_err = 0
    if self.window_full:
      anchor_depth_inds = (D_kf_d[0][0], D_kf_d[0][1])
      d = anchor_depth_inds[1] - anchor_depth_inds[0]
      first_depth_prior_err = linearize_multi_scalar_prior(self.sparse_log_depth[0,:d,:], self.sparse_log_depth_anchor[0,:d,:], H, g, anchor_depth_inds, sigma=self.cfg["sigmas"]["scale_prior"])

    total_err = mean_sq_photo_err + depth_prior_err1 + pose_prior_err + scale_prior_err + aff_scale_err + aff_bias_err

    # print("Mapping iter: ", self.iter, " error: ", total_err.item())

    # Solve and update
    # torch.cuda.synchronize()
    # start = time.time()
    delta = solve_system(H, g)
    # print((delta-delta2).flatten())
    # torch.cuda.synchronize()
    # end = time.time()
    # print("solve time: ", end-start)

    self.poses, self.mean_log_depths, self.sparse_log_depth, self.aff_params, \
        self.recent_poses, self.recent_aff_params = \
        update_vars(delta, self.poses, self.mean_log_depths, self.sparse_log_depth, self.aff_params,
            self.recent_poses, self.recent_aff_params, depth_inds)

    # TODO: Evaluate convergence
    self.iter += 1
    delta_norm = torch.norm(delta)
    abs_decrease = (self.total_err_prev - total_err)
    rel_decrease = abs_decrease/self.total_err_prev
    # print(delta_norm, abs_decrease, rel_decrease)
    if self.iter >= self.cfg["term_criteria"]["max_iter"] \
        or delta_norm < self.cfg["term_criteria"]["delta_norm"] \
        or abs_decrease < self.cfg["term_criteria"]["abs_tol"] \
        or rel_decrease < self.cfg["term_criteria"]["rel_tol"]:
      self.converged = True

    self.total_err_prev = total_err

    return self.converged