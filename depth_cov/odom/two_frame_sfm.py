import torch

from depth_cov.odom.odom_geometry import predict_log_depth, log_depth_to_depth, projection, backprojection, transformPoints
from depth_cov.odom.photo_utils import img_interp, setup_test_coords
import depth_cov.odom.robust_loss as robust
from depth_cov.utils.image_processing import ImagePyramidModule, IntrinsicsPyramidModule
import depth_cov.utils.lie_algebra as lie
from depth_cov.utils.utils import normalize_coordinates, normalize_coordinates_A, swap_coords_xy, unnormalize_coordinates

# Coarse-to-fine where inputs are lists except Tji_init, sparse_log_depth_init
def two_frame_sfm_pyr(Tji_init, sparse_log_depth_init, aff_init, mean_log_depth_init, test_coords_i, vals_i, Knm_Kmminv, img_and_grads_j, dr_prior_dd, H_prior_d_d, intrinsics, sigmas, term_criteria):  
  Tji = Tji_init.clone()
  sparse_log_depth = sparse_log_depth_init.clone()
  aff = aff_init.clone()
  mean_log_depth = mean_log_depth_init.clone()

  num_levels = len(vals_i)
  for l in range(num_levels):
    Tji, sparse_log_depth, aff, coords_j, depths_j = two_frame_sfm(Tji, sparse_log_depth, aff, mean_log_depth, test_coords_i[l], vals_i[l], Knm_Kmminv[l], img_and_grads_j[l], dr_prior_dd, H_prior_d_d, intrinsics[l], sigmas, term_criteria)

  return Tji, sparse_log_depth, aff, coords_j, depths_j

def setup_reference(img_and_grads, sparse_coords_norm, model, gaussian_covs, cov_level, intrinsics):
  c3 = img_and_grads[-1].shape[-3]
  c = c3//3
  device = img_and_grads[-1].device
  dtype = img_and_grads[-1].dtype

  intrinsics_pyr_module = IntrinsicsPyramidModule(0, len(img_and_grads), device)
  intrinsics_pyr = intrinsics_pyr_module(intrinsics, [1.0, 1.0])

  L_mm, E_m = model.get_covariance_chol(gaussian_covs, cov_level, sparse_coords_norm)
  dr_prior_dd, H_prior_d_d = linearize_sparse_depth_prior(L_mm)

  vals_pyr = []
  test_coords_pyr = []
  Knm_Kmminv_pyr = []
  img_sizes_pyr = []
  for i in range(len(img_and_grads)):
    img = img_and_grads[i][:,:c,:,:]
    img_grads = img_and_grads[i][:,c:,:,:]
    test_coords = setup_test_coords(img_grads, depth=None, grad_mag_thresh=None)

    test_coords_norm = normalize_coordinates(test_coords.to(dtype=dtype), img.shape[-2:])
    test_coords_pyr.append(test_coords)

    vals = img[:,:,test_coords[0,:,0],test_coords[0,:,1]]
    vals_pyr.append(vals)

    K_mn = model.get_cross_covariance(gaussian_covs, cov_level, sparse_coords_norm, test_coords_norm)
    Kmminv_Kmn = torch.cholesky_solve(K_mn, L_mm, upper=False)
    Knm_Kmminv = torch.transpose(Kmminv_Kmn, dim0=-2, dim1=-1)
    Knm_Kmminv_pyr.append(Knm_Kmminv)

    img_sizes_pyr.append(img.shape[-2:])

  return vals_pyr, test_coords_pyr, Knm_Kmminv_pyr, img_sizes_pyr, intrinsics_pyr, dr_prior_dd, H_prior_d_d

def linearize_sparse_depth_prior(L_mm):
  B, N_train, _ = L_mm.shape
  device = L_mm.device
  
  da_dd = torch.eye(N_train,device=device).unsqueeze(0).repeat(B,1,1)
  dr_dd = torch.linalg.solve_triangular(L_mm, da_dd, upper=False)
  # Note: This is K_mm_inv
  H_d_d = torch.einsum('hjk,hjl->hkl', dr_dd, dr_dd)

  return dr_dd, H_d_d

def linearize_mean_depth_prior_system(Knm_Kmminv):
  B, N, M = Knm_Kmminv.shape

  dr_dd = torch.sum(Knm_Kmminv, dim=(1), keepdim=True)/N
  H_d_d = torch.einsum('hjk,hjl->hkl', dr_dd, dr_dd)

  return dr_dd, H_d_d

def construct_sparse_depth_prior_system(sparse_log_depth_ref, mean_log_depth, H, g, dr_dd, H_d_d):
  # h(x) - z with z = 0
  a = sparse_log_depth_ref - mean_log_depth
  r = torch.matmul(dr_dd, a)
  total_err = torch.sum(torch.square(r), dim=(1,2))
  # Gradient g = -Jt @ r
  g[8:] -= torch.sum(dr_dd * r, dim=(1)).flatten().squeeze(0)
  # H = Jt @ J
  H[8:,8:] += H_d_d.squeeze(0)

  return total_err

def construct_mean_depth_prior_system(log_depth, mean_log_depth, Knm_Kmminv, H, g, dr_dd, H_d_d, sigma=1e-2):
  info_sqrt = 1.0/sigma
  info = info_sqrt * info_sqrt

  # h(x) - z with z = 0
  r = torch.mean(log_depth - mean_log_depth, dim=(1,2), keepdim=True)
  r *= info_sqrt
  total_err = torch.sum(torch.square(r), dim=(1,2))
  # Gradient g = -Jt @ r
  g[8:] -= torch.sum(info_sqrt * dr_dd * r, dim=(1)).flatten()
  # H = Jt @ J
  H[8:,8:] += info * torch.block_diag(*H_d_d)

  return total_err

def linearize_photo(Tji, vals_i, Pi, aff_params, img_and_grads_j, intrinsics):
  c = vals_i.shape[1]

  Pj, dPj_dTji, dPj_dPi = transformPoints(Tji, Pi)
  pj, dpj_dPj = projection(intrinsics, Pj)

  A_norm = 1.0/torch.as_tensor((img_and_grads_j.shape[-1], img_and_grads_j.shape[-2]), device=img_and_grads_j.device)

  # test_coords_target = swap_coords_xy(pj)
  # test_coords_target_norm = normalize_coordinates(test_coords_target, dims=img_and_grads_j.shape[-2:])
  pj_norm = normalize_coordinates_A(pj, A_norm)
  img_and_grads_interp, valid_mask = img_interp(img_and_grads_j, pj_norm)
  valid_mask = torch.logical_and(valid_mask, Pj[...,2] > 0)
  invalid_mask = torch.logical_not(valid_mask)
  vals_j = img_and_grads_interp[:,0:c,...]
  dIj_dw = torch.stack((img_and_grads_interp[:,c:2*c,...], img_and_grads_interp[:,2*c:3*c,...]), dim=3)
  dIj_dw = torch.permute(dIj_dw, (0,2,1,3))

  tmp = torch.exp(-aff_params[:,0,:]) * vals_i
  vals_i_new = tmp + aff_params[:,1,:]
  
  dI_daff = -torch.stack((-tmp, torch.ones_like(tmp)), dim=-1)
  dI_daff = torch.permute(dI_daff, (0,2,1,3))

  # Residuals  h(x) - z
  r = vals_j - vals_i_new
  r = torch.permute(r, (0,2,1))

  # Jacobians for pose and points
  dIj_dPj = torch.matmul(dIj_dw, dpj_dPj)
  dIj_dTji = torch.matmul(dIj_dPj, dPj_dTji)
  dIj_dPi = torch.matmul(dIj_dPj, dPj_dPi)

  test_coords_target = swap_coords_xy(pj)
  coords_j = test_coords_target[0:1,valid_mask[0,:],:]
  depths_j = Pj[0:1,valid_mask[0,:],2:3]

  return r, invalid_mask, dIj_dTji, dIj_dPi, dI_daff, coords_j, depths_j

def fill_photo_system(H, g, r, dIj_dTji, dIj_dd, dI_daff):
  g[:6] += -torch.sum(dIj_dTji * r[...,None], dim=(1,2)).squeeze(0)
  g[6:8] += -torch.sum(dI_daff * r[...,None], dim=(1,2)).squeeze(0)
  g[8:] += -torch.sum(dIj_dd * r[...,None], dim=(1,2)).squeeze(0)

  # Diagonal
  H[:6,:6] += torch.einsum('hijk,hijl->hkl', dIj_dTji, dIj_dTji).squeeze(0)
  H[6:8,6:8] += torch.einsum('hijk,hijl->hkl', dI_daff, dI_daff).squeeze(0)
  H[8:,8:] += torch.einsum('hijk,hijl->hkl', dIj_dd, dIj_dd).squeeze(0)
  # Off-diagonal
  H_Tji_d = torch.einsum('hijk,hijl->hkl', dIj_dTji, dIj_dd).squeeze(0)
  H[:6,8:] += H_Tji_d
  H[8:,:6] += torch.transpose(H_Tji_d, dim0=0, dim1=1)

  H_Tji_aff = torch.einsum('hijk,hijl->hkl', dIj_dTji, dI_daff).squeeze(0)
  H[:6,6:8] += H_Tji_aff
  H[6:8,:6] += torch.transpose(H_Tji_aff, dim0=0, dim1=1)

  H_aff_d = torch.einsum('hijk,hijl->hkl', dI_daff, dIj_dd).squeeze(0)
  H[6:8,8:] += H_aff_d
  H[8:,6:8] += torch.transpose(H_aff_d, dim0=0, dim1=1)

def construct_photo_system(
    Tji, sparse_log_depth, aff, mean_log_depth, 
    test_coords_i, vals_i, Knm_Kmminv, 
    img_and_grads_j, intrinsics, photo_sigma, H, g):
  
  # Reference points
  log_depth_i, dlogz_dd, _ = predict_log_depth(sparse_log_depth, mean_log_depth, Knm_Kmminv)
  z_i, dz_dlogz = log_depth_to_depth(log_depth_i)
  pi = swap_coords_xy(test_coords_i)
  Pi, dPi_dz = backprojection(intrinsics, pi, z_i)
  dPi_dlogz = torch.matmul(dPi_dz, dz_dlogz)
  dPi_dd = torch.matmul(dPi_dlogz, dlogz_dd)

  r, invalid_mask, dIj_dTji, dIj_dPi, dI_daff, coords_j, depths_j = linearize_photo(
      Tji, vals_i, Pi, aff, img_and_grads_j, intrinsics)

  dIj_dd = torch.matmul(dIj_dPi, dPi_dd)

  total_err = robustify_photo(r, invalid_mask, dIj_dTji, dIj_dd, dI_daff, photo_sigma)

  fill_photo_system(H, g, r, dIj_dTji, dIj_dd, dI_daff)

  return total_err, log_depth_i, coords_j, depths_j

def robustify_photo(r, invalid_mask, dIj_dTji, dIj_dd, dI_daff, sigma):
  info_sqrt = 1.0/sigma
  whitened_r = r*info_sqrt
  weight = robust.huber(whitened_r)
  weight[invalid_mask[...],:] = 0.0
  weight_sqrt = torch.sqrt(weight)

  # frame_err = torch.sum(rho)
  frame_err = torch.sum(torch.square(weight_sqrt*whitened_r))
  num_valid = invalid_mask.shape[-1] - torch.count_nonzero(invalid_mask, dim=-1)
  mean_err = frame_err/num_valid
  # print("Total error: ", torch.sum(torch.square(weight_sqrt*whitened_r)).item(), " Mean err: ", mean_err.item(), " Num valid: ", num_valid.item())
  
  r *= info_sqrt * weight_sqrt
  dIj_dTji *= info_sqrt * weight_sqrt[...,None]
  dIj_dd *= info_sqrt * weight_sqrt[...,None]
  dI_daff *= info_sqrt * weight_sqrt[...,None]

  # total_err = torch.sum(rho)
  total_err = torch.sum(torch.square(weight_sqrt*whitened_r))
  return total_err

def solve_delta(H, g):
  # U, S, Vh = torch.linalg.svd(H)
  # L = torch.linalg.cholesky(H, upper=False)
  L, _ = torch.linalg.cholesky_ex(H, upper=False, check_errors=False)
  delta = torch.cholesky_solve(g[:,None], L, upper=False)
  return delta

def update_vars(T, sparse_log_depth, aff, delta):
  delta_T = delta[:6,0].unsqueeze(0)
  T_new = lie.batch_se3(T, delta_T)

  delta_aff = delta[6:8]
  aff_new = aff + delta_aff

  delta_d = delta[8:]
  sparse_log_depth_new = sparse_log_depth + delta_d

  return T_new, sparse_log_depth_new, aff_new


def two_frame_sfm(
    Tji_init, sparse_log_depth_init, aff_init, mean_log_depth, 
    test_coords_i, vals_i, Knm_Kmminv, img_and_grads_j, 
    dr_prior_dd, H_prior_d_d,
    intrinsics, sigmas, term_criteria):

  device = Tji_init.device
  dtype = Tji_init.dtype

  N_train = sparse_log_depth_init.shape[1]
  D = 6 + 2 + N_train # Pose, aff, N points

  Tji = Tji_init.clone()
  sparse_log_depth = sparse_log_depth_init.clone()
  aff = aff_init.clone()

  # Precomputate linearizations
  dr_mean_dd, H_mean_d_d = linearize_mean_depth_prior_system(Knm_Kmminv)

  iter = 0
  done = False
  total_err_prev = float("inf")
  while not done:
    H = torch.zeros((D,D), device=device, dtype=dtype)
    g = torch.zeros((D), device=device, dtype=dtype)

    photo_err, log_depth, coords_j, depths_j = construct_photo_system(Tji, sparse_log_depth, aff, mean_log_depth, test_coords_i, vals_i, 
        Knm_Kmminv, img_and_grads_j, intrinsics, sigmas["photo"], H, g)

    depth_prior_err = construct_sparse_depth_prior_system(
      sparse_log_depth, mean_log_depth, H, g, dr_prior_dd, H_prior_d_d)
    mean_depth_err = construct_mean_depth_prior_system(
      log_depth, mean_log_depth, Knm_Kmminv, H, g, dr_mean_dd, H_mean_d_d, sigma=sigmas["mean_depth_prior"])

    total_err = photo_err + depth_prior_err + mean_depth_err
    delta = solve_delta(H, g)   
    Tji_new, sparse_log_depth_new, aff_new = update_vars(Tji, sparse_log_depth, aff, delta)
    Tji = Tji_new
    sparse_log_depth = sparse_log_depth_new

    iter += 1
    delta_norm = torch.norm(delta[:6])
    abs_decrease = total_err_prev - total_err
    rel_decrease = abs_decrease/total_err_prev
    if iter >= term_criteria["max_iter"] \
        or delta_norm < term_criteria["delta_norm"] \
        or abs_decrease < term_criteria["abs_tol"] \
        or rel_decrease < term_criteria["rel_tol"]:
      done = True
    
    total_err_prev = total_err

  return Tji, sparse_log_depth, aff, coords_j, depths_j