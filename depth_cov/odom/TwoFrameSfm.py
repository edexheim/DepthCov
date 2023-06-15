import torch
import torchvision.transforms.functional as TF

from depth_cov.core.samplers import sample_sparse_coords_norm
from depth_cov.odom.two_frame_sfm import setup_reference, two_frame_sfm_pyr
from depth_cov.utils.utils import fill_image, normalize_coordinates, str_to_dtype
from depth_cov.utils.image_processing import ImagePyramidModule, ImageGradientModule

class TwoFrameSfm():
  def __init__(self, cfg, intrinsics, model, cov_level, network_size):
    self.cfg = cfg
    self.device = cfg["device"]
    self.dtype = str_to_dtype(cfg["dtype"])

    self.intrinsics = intrinsics
    self.model = model
    self.cov_level = cov_level
    self.network_size = network_size

    self.has_reference = False
    self.is_init = False

    self.pose_init = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0)
    self.aff_init = torch.zeros((1,2,1), device=self.device, dtype=self.dtype)
    self.mean_log_depth_init = 0.0*torch.ones((1,1,1), device=self.device, dtype=self.dtype)

  def handle_frame(self, rgb):

    is_init = False
    img_and_grads = self.get_img_gradient_pyr(rgb)

    if not self.has_reference: # Init frame
      self.init_frame(rgb, img_and_grads)
      return False, None, None, None, None, None, None
    else: # Optimize reference
      T_curr_kf, sparse_log_depth_kf, aff_curr_kf, coords_curr, depth_curr = self.align_frame(img_and_grads)
      reproj_depth = fill_image(coords_curr, depth_curr, img_and_grads[-1].shape[-2:])
      
      kf_dist = torch.linalg.norm(T_curr_kf[:,:3,3])
      if torch.count_nonzero(~torch.isnan(reproj_depth)) < self.cfg["init"]["kf_num_pixels_viz"]:
        self.has_reference = False
      elif kf_dist > self.cfg["init"]["kf_depth_motion_ratio"] * torch.median(depth_curr):
        is_init = True

      return is_init, img_and_grads, T_curr_kf, aff_curr_kf, sparse_log_depth_kf, coords_curr, depth_curr

  def get_img_gradient_pyr(self, rgb):
    img_tracking = TF.rgb_to_grayscale(rgb)

    c, h, w = img_tracking.shape[-3:]
    img_pyr_module = ImagePyramidModule(c, 
        self.cfg["init"]["start_level"], self.cfg["init"]["end_level"], 
        self.device, self.dtype)
    img_pyr = img_pyr_module(img_tracking)

    gradient_module = ImageGradientModule(channels=c, device=self.device, dtype=self.dtype)
    img_and_grads = []
    for l in range(len(img_pyr)):
      gx, gy = gradient_module(img_pyr[l])
      img_and_grads_level = torch.cat((img_pyr[l], gx, gy), dim=1)
      img_and_grads.append(img_and_grads_level)
    
    return img_and_grads

  def init_frame(self, rgb_in, img_and_grads):
    h, w = rgb_in.shape[-2:]

    self.rgb = rgb_in
    self.img_and_grads = img_and_grads

    # Gaussian covs
    with torch.no_grad():
      rgb_r = TF.resize(self.rgb, self.network_size.tolist(), interpolation=TF.InterpolationMode.BILINEAR, antialias=True).float()
      with torch.no_grad():
        self.gaussian_covs = self.model(rgb_r)
        # TODO: Should we convert gaussian covs to double? Or handle float most places?
        for l in range(len(self.gaussian_covs)):
          self.gaussian_covs[l] = self.gaussian_covs[l].to(dtype=self.dtype)

      # Select sparse coords
      self.sparse_coords_norm = sample_sparse_coords_norm(self.gaussian_covs, 
        self.cfg["sampling"]["max_samples"], mode=self.cfg["sampling"]["mode"],
        max_stdev_thresh=self.cfg["sampling"]["max_stdev_thresh"], 
        model=self.model, model_level=self.cov_level)

      # Prep variables for two frame sfm
      self.vals_pyr, self.test_coords_pyr, self.Knm_Kmminv_pyr, self.img_sizes_pyr, \
          self.intrinsics_pyr, self.dr_prior_dd, self.H_prior_d_d \
          = setup_reference(
            self.img_and_grads, self.sparse_coords_norm, 
            self.model, self.gaussian_covs, self.cov_level,
            self.intrinsics)

    # Prep variables for keyframe initialization
    self.sparse_log_depth = self.mean_log_depth_init*torch.ones((1,self.sparse_coords_norm.shape[1],1), device=self.device, dtype=self.dtype)

    # Relative frame
    self.T_curr_kf = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0)
    self.aff_curr_kf = torch.zeros((1,2,1), device=self.device, dtype=self.dtype)

    self.has_reference = True

  def align_frame(self, img_and_grads):
    for i in range(len(img_and_grads)):
      img_and_grads[i] = img_and_grads[i].to(self.device)

    T_curr_kf, sparse_log_depth_kf, aff_kf, coords_curr, depth_curr  = two_frame_sfm_pyr(
      self.T_curr_kf, self.sparse_log_depth, self.aff_curr_kf, self.mean_log_depth_init, 
      self.test_coords_pyr, self.vals_pyr, self.Knm_Kmminv_pyr,
      img_and_grads, self.dr_prior_dd, self.H_prior_d_d,
      self.intrinsics_pyr, self.cfg["sigmas"], self.cfg["term_criteria"])

    return T_curr_kf, sparse_log_depth_kf, aff_kf, coords_curr, depth_curr

  # TODO: Check everything deleted!
  def delete_init_reference(self):
    del self.rgb
    del self.gaussian_covs
    del self.img_and_grads
    del self.sparse_coords_norm
    del self.vals_pyr
    del self.test_coords_pyr
    del self.Knm_Kmminv_pyr
    del self.img_sizes_pyr
    del self.intrinsics_pyr
    del self.dr_prior_dd
    del self.H_prior_d_d
    del self.sparse_log_depth
    del self.T_curr_kf
    del self.aff_curr_kf
    torch.cuda.empty_cache()

    self.has_reference = False