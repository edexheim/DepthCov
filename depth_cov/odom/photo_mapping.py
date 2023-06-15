import torch

import time

from depth_cov.utils.image_processing import ImageGradientModule
from depth_cov.odom.odom_geometry import predict_log_depth, predict_log_depth_img, log_depth_to_depth, depth_to_log_depth, projection, backprojection, transformPoints, between
from depth_cov.odom.odom_opt_utils import get_depth_inds, get_forward_edges, get_backward_edges, get_gradient, get_hessian_diag_block, get_hessian_off_diag_block, accumulate_gradient, accumulate_hessian_diag, accumulate_hessian_off_diag, get_dims
import depth_cov.odom.robust_loss as robust
from depth_cov.utils.utils import get_coord_img, swap_coords_xy, normalize_coordinates_A

def get_pose_pairs(poses1, median_depths1, poses2, radius, mode):
  dists = torch.cdist(poses1[:,:3,3], poses2[:,:3,3], compute_mode='use_mm_for_euclid_dist_if_necessary')
  
  if mode == "nearest":
    inds2 = torch.arange(dists.shape[1], device=poses1.device, dtype=torch.long)
    min_dists, inds1 = torch.min(dists, dim=0)
  elif mode == "radius":
    valid = dists < (radius * median_depths1)
    inds1, inds2 = torch.nonzero(valid, as_tuple=True)
  elif mode == "nearest_and_radius":
    # Ensure at least nearest is included, and then others from radius as well while avoiding duplicates
    inds2_nearest = torch.arange(dists.shape[1], device=poses1.device, dtype=torch.long)
    min_dists, inds1_nearest = torch.min(dists, dim=0)
    # Mask out nearest for remaining
    dists[inds1_nearest, inds2_nearest] = 1.0e8
    valid = dists < (radius * median_depths1)
    inds1_radius, inds2_radius = torch.nonzero(valid, as_tuple=True)
    inds1 = torch.cat((inds1_nearest, inds1_radius))
    inds2 = torch.cat((inds2_nearest, inds2_radius))
  else:
    raise ValueError("get_pose_pairs mode: " + mode + " is not implemented.")

  return inds1, inds2

def get_kf_edges(poses, median_depths, radius, D_kf_s, D_kf_d):
  inds1, inds2 = get_pose_pairs(poses, median_depths, poses, radius=radius, mode="radius")
  # Avoid pose with itself and consecutive keyframes)
  valid_pairs = torch.abs(inds1-inds2) > 1
  inds1 = inds1[valid_pairs].tolist()
  inds2 = inds2[valid_pairs].tolist()

  ref_ids = [b for b in inds1]
  target_ids = [b for b in inds2]
  ref_pose_inds = [(8*b, 8*b + 8) for b in inds1]
  target_pose_inds = [(8*b, 8*b + 8) for b in inds2]
  ref_scale_inds = [D_kf_s[b] for b in inds1]
  target_scale_inds = [D_kf_s[b] for b in inds2]
  ref_depth_inds = [D_kf_d[b] for b in inds1]
  target_depth_inds = [D_kf_d[b] for b in inds2]

  return ref_ids, target_ids, ref_pose_inds, target_pose_inds, \
      ref_scale_inds, target_scale_inds, ref_depth_inds, target_depth_inds

def get_one_way_edges(poses, median_depths, recent_poses, radius, D_kf_s, D_kf_d):
  B = poses.shape[0]

  if radius > 0.0:
    one_way_ref_inds, one_way_inds = get_pose_pairs(poses, median_depths, recent_poses, radius=radius, mode="nearest_and_radius")
  else:
    one_way_ref_inds, one_way_inds = get_pose_pairs(poses, None, recent_poses, radius=None, mode="nearest")
  one_way_ref_inds = one_way_ref_inds.tolist()
  one_way_inds = one_way_inds.tolist()

  one_way_ref_ids = [b for b in one_way_ref_inds]
  one_way_ids = [b for b in one_way_inds]
  ref_inds = [(8*kf_r, 8*kf_r + 8) for kf_r in one_way_ref_inds]
  target_inds = [(8*B + 8*r, 8*B + 8*r + 8) for r in one_way_inds]
  scale_inds = [D_kf_s[kf_r] for kf_r in one_way_ref_inds]
  sys_inds_di = [D_kf_d[kf_r] for kf_r in one_way_ref_inds]

  return one_way_ref_ids, one_way_ids, ref_inds, target_inds, scale_inds, sys_inds_di


def linearize_transformation(poses_i, Pi, poses_j):
  Tji, dTji_dTi, dTji_dTj = between(poses_i, poses_j)
  Pj, dPj_dTji, dPj_dPi = transformPoints(Tji, Pi)
  dPj_dTi = torch.matmul(dPj_dTji, dTji_dTi[:,None,:,:])
  dPj_dTj = torch.matmul(dPj_dTji, dTji_dTj[:,None,:,:])
  return Pj, dPj_dTi, dPj_dTj, dPj_dPi

def linearize_projection(Pj, intrinsics, A_norm):
  pj, dpj_dPj = projection(intrinsics, Pj)
  pj_norm = normalize_coordinates_A(pj, A_norm)
  return pj_norm, dpj_dPj

def linearize_image_space(img_and_grads_j, coords_norm):
  device = img_and_grads_j.device
  c3,h,w = img_and_grads_j.shape[1:4]
  c = c3//3

  valid_mask = (torch.abs(coords_norm) <= 0.99)
  valid_mask = torch.all(valid_mask, dim=-1)

  # Interpolate values
  img_and_grads_interp = torch.nn.functional.grid_sample(img_and_grads_j, coords_norm.unsqueeze(2), 
      mode="bilinear", padding_mode='zeros', align_corners=False)
  img_and_grads_interp = img_and_grads_interp.squeeze(3)

  vals_target = torch.permute(img_and_grads_interp[:,0:c,...], (0,2,1))
  dIt_dw = torch.stack((img_and_grads_interp[:,c:2*c,...], img_and_grads_interp[:,2*c:,...]), dim=3)
  dIt_dw = torch.permute(dIt_dw, (0,2,1,3))

  return vals_target, dIt_dw, valid_mask

def linearize_depth(target_log_depth_img_and_grads, coords_norm):
  device = target_log_depth_img_and_grads.device

  valid_mask = (torch.abs(coords_norm) <= 0.99)
  valid_mask = torch.all(valid_mask, dim=-1)

  # Interpolate values
  log_depth_and_grads_interp = torch.nn.functional.grid_sample(target_log_depth_img_and_grads, coords_norm.unsqueeze(2), 
      mode="nearest", padding_mode='zeros', align_corners=False)

  log_depth_and_grads_interp = torch.permute(log_depth_and_grads_interp, (0, 2, 3, 1))

  log_depth_target = log_depth_and_grads_interp[:,:,:,0]
  dlogDt_dw = log_depth_and_grads_interp[:,:,:,1:3]
  dlogDt_ds = log_depth_and_grads_interp[:,:,:,3:4]
  dlogDt_dd = log_depth_and_grads_interp[:,:,:,4:]

  return log_depth_target, dlogDt_dw, dlogDt_ds, dlogDt_dd, valid_mask

def robustify_photo_system_inplace(r, dIt_dw, dI_daffi, dI_daffj, invalid_mask, sigma):
  # Noise parameters
  info_sqrt = 1.0/sigma

  whitened_r = r*info_sqrt
  weight = robust.huber(whitened_r)
  weight[invalid_mask,...] = 0.0
  weight_sqrt = torch.sqrt(weight)

  r *= info_sqrt * weight_sqrt
  dIt_dw *= info_sqrt * weight_sqrt[...,None]
  dI_daffi *= info_sqrt * weight_sqrt[...,None]
  dI_daffj *= info_sqrt * weight_sqrt[...,None]

  total_err = torch.sum(torch.square(weight_sqrt*whitened_r))
  return total_err

def robustify_depth_system_inplace(r, dr_dTi, dr_dTj, dr_dsi, dr_ddi, dr_dsj, dr_ddj, invalid_mask, sigma):
  # Noise parameters
  info_sqrt = 1.0/sigma

  whitened_r = r*info_sqrt
  weight = robust.tukey(whitened_r)
  weight[invalid_mask,...] = 0.0
  weight_sqrt = torch.sqrt(weight)

  r *= info_sqrt * weight_sqrt
  dr_dTi *= info_sqrt * weight_sqrt[...,None]
  dr_dTj *= info_sqrt * weight_sqrt[...,None]
  dr_dsi *= info_sqrt * weight_sqrt[...,None]
  dr_ddi *= info_sqrt * weight_sqrt[...,None]
  dr_dsj *= info_sqrt * weight_sqrt[...,None]
  dr_ddj *= info_sqrt * weight_sqrt[...,None]

  total_err = torch.sum(torch.square(weight_sqrt*whitened_r))
  return total_err

# When dr_dx is 2*B batches of same type so can batch all together and accumulate
def accumulate_self_terms(dr_dx, r_dup, sys_inds_x1, sys_inds_x2, H, g):
  B = dr_dx.shape[0]//2

  d1 = []
  d2 = []
  for b in range(B):
    d1.append(sys_inds_x1[b][1] - sys_inds_x1[b][0])
    d2.append(sys_inds_x2[b][1] - sys_inds_x2[b][0])

  grad_batch = get_gradient(dr_dx, r_dup)
  for b in range(B):
    accumulate_gradient(grad_batch[b,:d1[b]], sys_inds_x1[b], g)
    accumulate_gradient(grad_batch[b+B,:d2[b]], sys_inds_x2[b], g)

  H_batch = get_hessian_diag_block(dr_dx)
  for b in range(B):
    accumulate_hessian_diag(H_batch[b,:d1[b],:d1[b]], sys_inds_x1[b], H)
    accumulate_hessian_diag(H_batch[b+B,:d2[b],:d2[b]], sys_inds_x2[b], H)

  H_off_diag_batch = get_hessian_off_diag_block(dr_dx[:B,...], dr_dx[B:,...])
  for b in range(B):
    accumulate_hessian_off_diag(H_off_diag_batch[b,:d1[b],:d2[b]], sys_inds_x1[b], sys_inds_x2[b], H)

def accumulate_cross_terms(dr_dx, dr_dy, sys_inds_x1, sys_inds_x2, 
    sys_inds_y1, sys_inds_y2, H, g):
  
  B = dr_dx.shape[0]//2

  H_off_diag_batch = get_hessian_off_diag_block(dr_dx[:B,...], dr_dy[:B,...])
  for b in range(B):
    d1 = sys_inds_x1[b][1] - sys_inds_x1[b][0]
    d2 = sys_inds_y1[b][1] - sys_inds_y1[b][0]
    accumulate_hessian_off_diag(H_off_diag_batch[b,:d1,:d2], sys_inds_x1[b], sys_inds_y1[b], H)

  H_off_diag_batch = get_hessian_off_diag_block(dr_dx[:B,...], dr_dy[B:,...])
  for b in range(B):
    d1 = sys_inds_x1[b][1] - sys_inds_x1[b][0]
    d2 = sys_inds_y2[b][1] - sys_inds_y2[b][0]
    accumulate_hessian_off_diag(H_off_diag_batch[b,:d1,:d2], sys_inds_x1[b], sys_inds_y2[b], H)

  H_off_diag_batch = get_hessian_off_diag_block(dr_dx[B:,...], dr_dy[:B,...])
  for b in range(B):
    d1 = sys_inds_x2[b][1] - sys_inds_x2[b][0]
    d2 = sys_inds_y1[b][1] - sys_inds_y1[b][0]
    accumulate_hessian_off_diag(H_off_diag_batch[b,:d1,:d2], sys_inds_x2[b], sys_inds_y1[b], H)

  H_off_diag_batch = get_hessian_off_diag_block(dr_dx[B:,...], dr_dy[B:,...])
  for b in range(B):
    d1 = sys_inds_x2[b][1] - sys_inds_x2[b][0]
    d2 = sys_inds_y2[b][1] - sys_inds_y2[b][0]
    accumulate_hessian_off_diag(H_off_diag_batch[b,:d1,:d2], sys_inds_x2[b], sys_inds_y2[b], H)


def batch_photometric_constraints(poses_i, aff_params_i, vals_i, P_i, dP_dsi, dP_ddi,
    poses_j, aff_params_j, img_and_grads_j, 
    intrinsics, A_norm,
    photo_ref_pose_aff_inds, photo_target_pose_aff_inds, photo_ref_scale_inds, photo_ref_depth_inds,
    sigma, H, g, total_err):
  
  device = poses_i.device
  dtype = poses_i.dtype

  B = poses_i.shape[0]

  Pj, dPj_dTi, dPj_dTj, dPj_dPi \
    = linearize_transformation(poses_i, P_i, poses_j)
  x_samples, dpj_dPj = linearize_projection(Pj, intrinsics, A_norm)

  dPj_dsi = torch.matmul(dPj_dPi, dP_dsi)
  dPj_ddi = torch.matmul(dPj_dPi, dP_ddi)
  
  vals_target, dIt_dw, valid_mask = linearize_image_space(img_and_grads_j, x_samples)
  invalid_mask = torch.logical_not(valid_mask)

  vals_i_scaled = torch.exp(aff_params_j[:,0:1,:] - aff_params_i[:,0:1,:]) * vals_i
  photo_bias = aff_params_j[:,1:2,:] - aff_params_i[:,1:2,:]
  r_photo = vals_target - vals_i_scaled + photo_bias

  # Affine Jacobians
  dI_daffi = torch.stack((vals_i_scaled, -torch.ones_like(vals_i_scaled)), dim=-1)
  dI_daffj = -dI_daffi

  # r_photo, dIt_dw, dI_daffi, dI_daffj modified
  photo_err = robustify_photo_system_inplace(r_photo, dIt_dw, dI_daffi, dI_daffj, invalid_mask, sigma=sigma)
  total_err += photo_err

  # Jacobians (keep transformations and affine parameters together)
  dIt_ij = torch.empty((2*B, vals_i.shape[1], vals_i.shape[2], 8), device=device, dtype=dtype)
  dIt_ij[:B,:,:,6:] = dI_daffi
  dIt_ij[B:,:,:,6:] = dI_daffj

  # Chain rule
  dIt_dPj = torch.matmul(dIt_dw, dpj_dPj)

  dIt_ij[:B,:,:,:6] = torch.matmul(dIt_dPj, dPj_dTi)
  dIt_ij[B:,:,:,:6] = torch.matmul(dIt_dPj, dPj_dTj)

  dIt_dsi = torch.matmul(dIt_dPj, dPj_dsi)
  dIt_ddi = torch.matmul(dIt_dPj, dPj_ddi)

  # Depth dims
  d_dim_list = []
  for b in range(B):
    d = photo_ref_depth_inds[b][1] - photo_ref_depth_inds[b][0]
    d_dim_list.append(d)

  torch.cuda.synchronize()
  start = time.time()

  # Pose
  r_dup = torch.cat((r_photo, r_photo), dim=0)
  accumulate_self_terms(dIt_ij, r_dup, photo_ref_pose_aff_inds, photo_target_pose_aff_inds, H, g)
  # Pose - scale
  H_off_diag_batch = get_hessian_off_diag_block(dIt_ij, torch.cat((dIt_dsi, dIt_dsi), dim=0))
  for b in range(B):
    accumulate_hessian_off_diag(H_off_diag_batch[b,:,:], photo_ref_pose_aff_inds[b], photo_ref_scale_inds[b], H)
    accumulate_hessian_off_diag(H_off_diag_batch[b+B,:,:], photo_target_pose_aff_inds[b], photo_ref_scale_inds[b], H)
  # Pose - depth
  H_off_diag_batch = get_hessian_off_diag_block(dIt_ij, torch.cat((dIt_ddi, dIt_ddi), dim=0))
  for b in range(B):
    d = d_dim_list[b]
    accumulate_hessian_off_diag(H_off_diag_batch[b,:,:d], photo_ref_pose_aff_inds[b], photo_ref_depth_inds[b], H)
    accumulate_hessian_off_diag(H_off_diag_batch[b+B,:,:d], photo_target_pose_aff_inds[b], photo_ref_depth_inds[b], H)
  
  # Scale
  grad_batch = get_gradient(dIt_dsi, r_photo)
  for b in range(B):
    accumulate_gradient(grad_batch[b,:], photo_ref_scale_inds[b], g)
  H_batch = get_hessian_diag_block(dIt_dsi)
  for b in range(B):
    accumulate_hessian_diag(H_batch[b,:,:], photo_ref_scale_inds[b], H)
  # Scale - depth
  H_off_diag_batch = get_hessian_off_diag_block(dIt_dsi, dIt_ddi)
  for b in range(B):
    d = d_dim_list[b]
    accumulate_hessian_off_diag(H_off_diag_batch[b,:,:d], photo_ref_scale_inds[b], photo_ref_depth_inds[b], H)

  # Depth
  grad_batch = get_gradient(dIt_ddi, r_photo)
  for b in range(B):
    d = d_dim_list[b]
    accumulate_gradient(grad_batch[b,:d], photo_ref_depth_inds[b], g)
  H_batch = get_hessian_diag_block(dIt_ddi)
  for b in range(B):
    d = d_dim_list[b]
    accumulate_hessian_diag(H_batch[b,:d,:d], photo_ref_depth_inds[b], H)
    
  # torch.cuda.synchronize()
  # end = time.time()
  # print("accumulate terms: ", end-start)

def batch_geometric_constraints(x_samples, depth_ref,
    dpj_dPj, dPj_dTi, dPj_dTj, dPj_dsi, dPj_ddi,
    target_log_depth_img_and_grads,
    sys_inds_dict, sigma, H, g, total_err):

  device = x_samples.device
  dtype = x_samples.dtype

  B, N, _, D = dPj_ddi.shape

  log_depth_target, dlogDt_dw, dr_dsj, dr_ddj, valid_mask = linearize_depth(target_log_depth_img_and_grads, x_samples)
  invalid_mask = torch.logical_not(valid_mask)

  # Depth difference
  log_depth_ref, dlogd_dz = depth_to_log_depth(depth_ref)
  r_log_depth = log_depth_target - log_depth_ref

  dr_dPj = torch.matmul(dlogDt_dw, dpj_dPj)
  dr_dPj[...,2:3] -=  dlogd_dz

  dr_dTi = torch.matmul(dr_dPj, dPj_dTi)
  dr_dTj = torch.matmul(dr_dPj, dPj_dTj)
  dr_dsi = torch.matmul(dr_dPj, dPj_dsi)
  dr_ddi = torch.matmul(dr_dPj, dPj_ddi)

  depth_err = robustify_depth_system_inplace(r_log_depth, dr_dTi, dr_dTj, dr_dsi, dr_ddi, dr_dsj, dr_ddj, invalid_mask, sigma=sigma)
  total_err += depth_err

  # Append variables together since symmetric between frames

  # Pad poses with affine for simplicity
  dr_dT_ij = torch.empty((2*B, N, 1, 8), device=device, dtype=dtype)
  dr_dT_ij[:B,:,:,:6] = dr_dTi
  dr_dT_ij[B:,:,:,:6] = dr_dTj
  dr_dT_ij[...,6:] = 0.0 # Affine parameters not involved

  dr_s_ij = torch.empty((2*B, N, 1, 1), device=device, dtype=dtype)
  dr_s_ij[:B,:,:,:] = dr_dsi
  dr_s_ij[B:,:,:,:] = dr_dsj

  dr_d_ij = torch.empty((2*B, N, 1, D), device=device, dtype=dtype)
  dr_d_ij[:B,:,:,:] = dr_ddi
  dr_d_ij[B:,:,:,:] = dr_ddj

  r_log_depth_dup = torch.cat((r_log_depth, r_log_depth), dim=0)

  ## Self terms - 6 diag blocks, 3 off-diag blocks
  accumulate_self_terms(dr_dT_ij, r_log_depth_dup, 
      sys_inds_dict["ref_pose_aff"], sys_inds_dict["target_pose_aff"], H, g)
  accumulate_self_terms(dr_s_ij, r_log_depth_dup, 
      sys_inds_dict["ref_scale"], sys_inds_dict["target_scale"], H, g)
  accumulate_self_terms(dr_d_ij, r_log_depth_dup, 
      sys_inds_dict["ref_depth"], sys_inds_dict["target_depth"], H, g)

  ## Cross terms - 12 off-diag blocks
  accumulate_cross_terms(dr_dT_ij, dr_s_ij, 
      sys_inds_dict["ref_pose_aff"], sys_inds_dict["target_pose_aff"], 
      sys_inds_dict["ref_scale"], sys_inds_dict["target_scale"],
      H, g)
  accumulate_cross_terms(dr_dT_ij, dr_d_ij, 
      sys_inds_dict["ref_pose_aff"], sys_inds_dict["target_pose_aff"], 
      sys_inds_dict["ref_depth"], sys_inds_dict["target_depth"],
      H, g)
  accumulate_cross_terms(dr_s_ij, dr_d_ij, 
      sys_inds_dict["ref_scale"], sys_inds_dict["target_scale"], 
      sys_inds_dict["ref_depth"], sys_inds_dict["target_depth"],
      H, g)


# B poses, img_and_grads
# B-1 vals_ref, img_and_grads, kernel matrices, test_coords_ref
def create_photo_system(poses, mean_log_depth, sparse_log_depth, aff_params, 
    recent_poses, recent_aff_params, recent_img_and_grads,
    img_and_grads, Knm_Kmminv, depth_inds, intrinsics, sigmas, photo_construction_cfg, graph_cfg, H, g):

  device = poses.device
  dtype = poses.dtype

  total_err = torch.tensor([0.0], device=device, dtype=dtype)
  A_norm = 1.0/torch.as_tensor((img_and_grads.shape[-1], img_and_grads.shape[-2]), device=img_and_grads.device, dtype=dtype)

  num_kf = poses.shape[0]
  num_recent = recent_poses.shape[0]
  D, D_kf_T, D_kf_aff, D_rec_T, D_rec_aff, D_kf_s, D_kf_d = get_dims(num_kf, num_recent, depth_inds)

  c3 = img_and_grads.shape[1]
  c = c3//3

  # Precomputation for depth maps (independent of projection to different frames)
  test_coord_img = get_coord_img(img_and_grads.shape[-2:], device, batch_size=num_kf)
  # Sampling pixels
  # TODO: Allow for random sampling or gradient-weighted sampling? Right now deterministic
  mean_sq_grad_norm = torch.sqrt(torch.square(img_and_grads[:,1,:,:]) + torch.square(img_and_grads[:,2,:,:]))
  max_mean_sq_grad_norm, max_indices = torch.nn.functional.max_pool2d(mean_sq_grad_norm, 
      kernel_size=photo_construction_cfg["nonmax_suppression_window"], return_indices=True)
  max_indices = max_indices.flatten(start_dim=1, end_dim=2)
  # Pad depth vars to max size to handle batching
  num_samples = max_indices.shape[1]
  depth_dim = sparse_log_depth.shape[1]
  # Form
  batch_inds = torch.arange(num_kf,device=device).unsqueeze(1).repeat(1,num_samples)
  test_coords_ref = test_coord_img.view(num_kf, -1, 2)[batch_inds, max_indices, :]
  vals_ref = torch.permute(img_and_grads[:,:c,...], (0,2,3,1)).view(num_kf, -1, c)[batch_inds, max_indices, :]
  Knm_Kmminv_ref = Knm_Kmminv.view(num_kf,-1,depth_dim)[batch_inds,max_indices,:]
    
  # Batch depth Jacobians
  log_depth_ref, dlogz_dd, dlogz_ds = predict_log_depth(
      sparse_log_depth, mean_log_depth, Knm_Kmminv_ref)
  z_ref, dz_dlogz = log_depth_to_depth(log_depth_ref)
  p = swap_coords_xy(test_coords_ref)
  P_ref, dP_dz = backprojection(intrinsics, p, z_ref)
  dP_dlogz = torch.matmul(dP_dz, dz_dlogz)
  dP_ds_ref= torch.matmul(dP_dlogz, dlogz_ds)
  dP_dd_ref = torch.matmul(dP_dlogz, dlogz_dd)

  B = num_kf
  R = num_recent

  ## Graph Construction  
  ref_ids_f, target_ids_f, ref_pose_inds_f, target_pose_inds_f, ref_scale_inds_f, target_scale_inds_f, ref_depth_inds_f, target_depth_inds_f = get_forward_edges(B, D_kf_s, D_kf_d)
  ref_ids_b, target_ids_b, ref_pose_inds_b, target_pose_inds_b, ref_scale_inds_b, target_scale_inds_b, ref_depth_inds_b, target_depth_inds_b = get_backward_edges(B, D_kf_s, D_kf_d)

  median_depths = torch.median(z_ref, dim=1, keepdim=False).values
  # Get keyframe edges within radius
  if graph_cfg["radius"] > 0.0:
    ref_ids_kf, target_ids_kf, ref_pose_inds_kf, target_pose_inds_kf, ref_scale_inds_kf, target_scale_inds_kf, ref_depth_inds_kf, target_depth_inds_kf = get_kf_edges(poses, median_depths, graph_cfg["radius"], D_kf_s, D_kf_d)
  else:
    ref_ids_kf, target_ids_kf, ref_pose_inds_kf, target_pose_inds_kf, ref_scale_inds_kf, target_scale_inds_kf, ref_depth_inds_kf, target_depth_inds_kf = [], [], [], [], [], [], [], []
  # Get one-way edges
  if R > 0:
    one_way_ref_ids, one_way_ids, ref_inds_r, target_inds_r, scale_inds_r, depth_inds_r = get_one_way_edges(poses, median_depths, recent_poses, graph_cfg["radius"], D_kf_s, D_kf_d)
  else:
    one_way_ref_ids, one_way_ids, ref_inds_r, target_inds_r, scale_inds_r, depth_inds_r = [], [], [], [], [], []


  kf_ref_ids = ref_ids_f + ref_ids_b + ref_ids_kf
  kf_target_ids = target_ids_f + target_ids_b + target_ids_kf

  # Not including recent frames until photometric dict formed
  ref_pose_inds = ref_pose_inds_f + ref_pose_inds_b + ref_pose_inds_kf
  target_pose_inds = target_pose_inds_f + target_pose_inds_b + target_pose_inds_kf
  ref_scale_inds = ref_scale_inds_f + ref_scale_inds_b + ref_scale_inds_kf
  ref_depth_inds = ref_depth_inds_f + ref_depth_inds_b + ref_depth_inds_kf

  # Include recent frames and references for photometric terms
  photo_ref_pose_aff_inds = ref_pose_inds + ref_inds_r
  photo_target_pose_aff_inds = target_pose_inds + target_inds_r
  photo_ref_scale_inds = ref_scale_inds + scale_inds_r
  photo_ref_depth_inds = ref_depth_inds + depth_inds_r

  all_ref_ids = kf_ref_ids + one_way_ref_ids

  torch.cuda.synchronize()
  start = time.time()

  num_kf_pairs = len(kf_target_ids)
  num_total_pairs = len(all_ref_ids)
  num_batches = 1 + (num_total_pairs-1) // photo_construction_cfg["pairwise_batch_size"]
  r_start = num_kf_pairs
  r_end = num_total_pairs
  for b in range(num_batches):
    b1 = b * photo_construction_cfg["pairwise_batch_size"]
    b2 = b1 + photo_construction_cfg["pairwise_batch_size"]

    batch_ref_ids = all_ref_ids[b1:b2]
    # Handle target vars since e-way need different handling
    if b2 <= num_kf_pairs:
      batch_kf_target_ids = kf_target_ids[b1:b2]
      target_poses = poses[batch_kf_target_ids]
      target_aff_params = aff_params[batch_kf_target_ids]
      target_img_and_grads = img_and_grads[batch_kf_target_ids]

      photo_target_inds = photo_target_pose_aff_inds[b1:b2]
    else:
      batch_kf_target_ids = kf_target_ids[b1:b2]
      r1 = max(b1, r_start) - r_start
      r2 = min(b2, r_end) - r_start
      recent_target_ids = one_way_ids[r1:r2]

      target_poses = torch.cat((poses[batch_kf_target_ids,...], recent_poses[recent_target_ids,...]), dim=0)
      target_aff_params = torch.cat((aff_params[batch_kf_target_ids,...], recent_aff_params[recent_target_ids,...]), dim=0)
      target_img_and_grads = torch.cat((img_and_grads[batch_kf_target_ids,...], recent_img_and_grads[recent_target_ids,...]), dim=0)
      
      photo_target_inds = photo_target_pose_aff_inds[b1:b2] + one_way_ids[r1:r2]

    batch_photometric_constraints(
        poses[batch_ref_ids], aff_params[batch_ref_ids], vals_ref[batch_ref_ids], P_ref[batch_ref_ids], dP_ds_ref[batch_ref_ids], dP_dd_ref[batch_ref_ids],
        target_poses, target_aff_params, target_img_and_grads,
        intrinsics, A_norm,
        photo_ref_pose_aff_inds[b1:b2], photo_target_inds, photo_ref_scale_inds[b1:b2], photo_ref_depth_inds[b1:b2],
        sigmas["photo"], H, g, total_err)
  
  # torch.cuda.synchronize()
  # end = time.time()
  # print("photo constraints: ", end-start)

  # if B >= 6:

  #   depth_sys_inds_dict = {
  #     "ref_pose_aff": ref_pose_inds,
  #     "target_pose_aff": target_pose_inds,
  #     "ref_scale": ref_scale_inds,
  #     "target_scale": target_scale_inds,
  #     "ref_depth": ref_depth_inds,
  #     "target_depth": target_depth_inds,
  #   }

  #   log_depth_img, dlogz_dsj, dlogz_ddj = predict_log_depth_img(
  #       sparse_log_depth, mean_log_depth, Knm_Kmminv)
  #   gradient_module = ImageGradientModule(channels=1, device=device, dtype=dtype)
  #   log_depth_img = torch.permute(log_depth_img, (0,3,1,2))
  #   gx, gy = gradient_module(log_depth_img)

  #   log_depth_img_and_grads = torch.cat((
  #     log_depth_img,
  #     gx,
  #     gy,
  #     torch.permute(dlogz_dsj[:,:,:,0,:], (0,3,1,2)),
  #     torch.permute(dlogz_ddj[:,:,:,0,:], (0,3,1,2)),
  #   ), dim=1)

  #   # Geometric factors require both to be keyframes, so only load beginning of photo vars
  #   n_kf_c = len(ref_ids)
  #   zj = Pj[:n_kf_c,:,2:3]
  #   batch_geometric_constraints(x_samples[:n_kf_c], zj,
  #       dpj_dPj[:n_kf_c], dPj_dTi[:n_kf_c], dPj_dTj[:n_kf_c], dPj_dsi[:n_kf_c], dPj_ddi[:n_kf_c],
  #       log_depth_img_and_grads[target_ids],
  #       depth_sys_inds_dict, sigmas["log_depth"], H, g, total_err)

  return total_err, [kf_ref_ids, kf_target_ids], [one_way_ref_ids, one_way_ids]
  