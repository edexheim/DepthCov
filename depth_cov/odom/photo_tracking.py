import torch

from depth_cov.odom.odom_geometry import skew_symmetric, transformPointsNoJ, projection, projectionNoJ, backprojection, transform_project
from depth_cov.odom.photo_utils import img_interp, setup_test_coords
import depth_cov.odom.robust_loss as robust
from depth_cov.utils.utils import swap_coords_xy, normalize_coordinates_A
from depth_cov.utils.image_processing import ImageGradientModule, ImagePyramidModule, DepthPyramidModule, IntrinsicsPyramidModule
import depth_cov.utils.lie_algebra as lie

from torch.profiler import profile, ProfilerActivity, record_function

# Coarse-to-fine where inputs are lists except Tji_init
def photo_tracking_pyr(Tji_init, aff_init, vals_i, Pi, dI_dT, intrinsics, img_j, photo_sigma, term_criteria):
  Tji = Tji_init.clone()
  aff = aff_init.clone()
  num_levels = len(vals_i)
  for l in range(num_levels):
    Tji, aff, coords_j, depths_j = photo_level_tracking(Tji, aff, vals_i[l], Pi[l], dI_dT[l], img_j[l], intrinsics[l], photo_sigma, term_criteria)

  return Tji, aff, coords_j, depths_j

# IC precalculate Jacobians at theta=0
def precalc_jacobians(dI_dw, P, vals, intrinsics):
  c = dI_dw.shape[1]
  device = dI_dw.device
  dtype = dI_dw.dtype

  _, n, _ = P.shape
  dPi_dT = torch.empty((n, 3, 6), device=device, dtype=dtype)
  # dPi_dT[:,:,3:] = torch.eye(3, device=device)[None,:,:]
  dPi_dT[:,:,3:] = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(n,1,1)
  dPi_dT[:,:,:3] = -skew_symmetric(P)

  _, dpi_dPi = projection(intrinsics, P)
  dpi_dPi = dpi_dPi
  dpi_dT = torch.matmul(dpi_dPi, dPi_dT)
  dI_dT = torch.matmul(dI_dw, dpi_dT)

  dI_dp = torch.cat((dI_dT, \
      torch.permute(vals, (0,2,1)).unsqueeze(-1), \
      torch.ones((dI_dT.shape[0], n, 1, 1), device=device, dtype=dtype)), dim=-1)

  return dI_dp

def setup_reference(img, depth, intrinsics, start_level, end_level, depth_interp_mode, grad_mag_thresh):
  c, h, w = img.shape[-3:]
  device = img.device
  dtype = img.dtype

  img_pyr_module = ImagePyramidModule(c, start_level, end_level, device)
  img_pyr = img_pyr_module(img)
  
  depth_pyr_module = DepthPyramidModule(start_level, end_level, depth_interp_mode, device)
  depth_pyr = depth_pyr_module(depth)

  intrinsics_pyr_module = IntrinsicsPyramidModule(start_level, end_level, device)
  intrinsics_pyr = intrinsics_pyr_module(intrinsics, [1.0, 1.0])

  grad_module = ImageGradientModule(channels=c, device=device)

  test_coords_pyr = []
  vals_pyr = []
  P_pyr = []
  dI_dT_pyr = []
  img_sizes_pyr = []
  for i in range(len(img_pyr)):
    grad_module = ImageGradientModule(channels=c, device=device)
    gx, gy = grad_module(img_pyr[i])
    img_grads = torch.cat((gx, gy), dim=1)

    test_coords = setup_test_coords(img_grads, depth=depth_pyr[i], grad_mag_thresh=grad_mag_thresh)

    vals = img_pyr[i][:,:,test_coords[0,:,0],test_coords[0,:,1]]
    depths = depth_pyr[i][:,:,test_coords[0,:,0],test_coords[0,:,1]]
    depths = torch.permute(depths, (0,2,1))

    gx = gx[0,:,test_coords[0,:,0],test_coords[0,:,1]]
    gy = gy[0,:,test_coords[0,:,0],test_coords[0,:,1]]
    dI_dw = torch.stack((gx, gy), dim=2)
    dI_dw = torch.permute(dI_dw, (1,0,2))

    test_coords_xy = swap_coords_xy(test_coords)
    P, _ = backprojection(intrinsics_pyr[i], test_coords_xy, depths)
    dI_dT = precalc_jacobians(dI_dw, P, vals, intrinsics_pyr[i])

    test_coords_pyr.append(test_coords)
    vals_pyr.append(vals)
    P_pyr.append(P)
    dI_dT_pyr.append(dI_dT)
    img_sizes_pyr.append(img_pyr[i].shape[-2:])

  return test_coords_pyr, vals_pyr, P_pyr, dI_dT_pyr, img_sizes_pyr, intrinsics_pyr

def robustify_photo(r, dIt_dT, invalid_mask, photo_sigma):
  info_sqrt = 1.0/photo_sigma
  whitened_r = r*info_sqrt
  weight = robust.huber(whitened_r)
  weight[invalid_mask[...],:] = 0.0

  total_err = torch.sum(weight*torch.square(whitened_r))
  num_valid = invalid_mask.shape[-1] - torch.count_nonzero(invalid_mask, dim=-1)
  mean_err = total_err/num_valid

  tmp = weight[0,:,0:1] * dIt_dT[0,:,0,:]
  grad = torch.sum(tmp*r[0,:,:], dim=0).unsqueeze(0)
  H = torch.matmul(tmp.T, dIt_dT[0,:,0,:]).unsqueeze(0)

  return H, grad, total_err, mean_err

def solve_delta(H, grad):
  L, _ = torch.linalg.cholesky_ex(H, upper=False, check_errors=False)
  delta = torch.cholesky_solve(grad[...,None], L, upper=False)
  return delta

# TODO: Batch
def update_pose_ic(T, aff, delta):
  delta_T = delta[:,:6,0]
  T_new = torch.matmul(T, lie.se3_exp(-delta_T))
  
  delta_a = delta[:,6,0]
  delta_b = delta[:,7,0]
  aff_new = torch.empty_like(aff)

  aff_new[:,0] = aff[:,0] - delta_a
  aff_new[:,1] = aff[:,1] - delta_b  

  return T_new, aff_new

def tracking_iter(Tji, Pi, intrinsics, img_j, aff, vals_i, dI_dT, photo_sigma, A_norm):
  pj, depth_j = transform_project(intrinsics, Tji, Pi)
  
  pj_norm = normalize_coordinates_A(pj, A_norm)
  
  vals_target, valid_mask = img_interp(img_j, pj_norm)
  valid_mask = torch.logical_and(valid_mask, depth_j[...,0] > 0)
  invalid_mask = torch.logical_not(valid_mask)

  tmp = torch.exp(-aff[:,None,0]) * vals_target
  dI_dT[...,6] = torch.permute(-tmp, (0,2,1))
  vals_target = tmp + aff[:,None,1]

  r = vals_target - vals_i
  r = torch.permute(r, (0,2,1))

  H, grad, total_err, mean_err = robustify_photo(r, dI_dT, invalid_mask, photo_sigma)

  # print("Tracking: ", iter, total_err.item(), mean_err.item())

  delta = solve_delta(H, grad)   
  Tji_new, aff_new = update_pose_ic(Tji, aff, delta)

  return Tji_new, aff_new, delta, mean_err, pj, valid_mask, depth_j

# Inverse compositional tracking
def photo_level_tracking(Tji_init, aff_init, vals_i, Pi, dI_dT, img_j, intrinsics, photo_sigma, term_criteria):
  Tji = Tji_init.clone()
  aff = aff_init.clone()

  A_norm = 1.0/torch.as_tensor((img_j.shape[-1], img_j.shape[-2]), device=img_j.device, dtype=img_j.dtype)

  iter = 0
  done = False
  mean_err_prev = float("inf")
  while not done:
    Tji, aff, delta, mean_err, p_j, valid_mask, depth_j = tracking_iter(Tji, Pi, intrinsics, img_j, aff, vals_i, dI_dT, photo_sigma, A_norm)

    # print("Tracking: ", iter, mean_err.item())

    iter += 1
    delta_norm = torch.norm(delta)
    abs_decrease = mean_err_prev - mean_err
    rel_decrease = abs_decrease/mean_err_prev
    if iter >= term_criteria["max_iter"] \
      or abs_decrease < term_criteria["abs_tol"] \
        or delta_norm < term_criteria["delta_norm"] \
        or rel_decrease < term_criteria["rel_tol"]:
      done = True
      # print(iter, abs_decrease, delta_norm, rel_decrease)
    mean_err_prev = mean_err

  test_coords_target = swap_coords_xy(p_j)
  coords_j = test_coords_target[0:1,valid_mask[0,:],:]
  depths_j = depth_j[0:1,valid_mask[0,:],:]

  return Tji, aff, coords_j, depths_j