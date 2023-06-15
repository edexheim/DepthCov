import torch

import depth_cov.utils.lie_algebra as lie

from typing import Tuple, List

def get_depth_inds(sparse_coords_norm_list):
  depth_dims = [depths.shape[1] for depths in sparse_coords_norm_list]
  depth_dims.insert(0,0)
  depth_inds = torch.cumsum(torch.tensor(depth_dims), dim=0)
  return depth_inds

def get_dims(num_keyframes, num_rec_frames, depth_inds):
  D = (6+2+1)*num_keyframes + depth_inds[-1] + (6+2)*num_rec_frames

  # Put T and aff for same pose together for efficiency
  D_kf_T = [(8*b, 8*b + 6) for b in range(num_keyframes)]
  D_kf_aff = [(D_kf_T[b][0] + 6, D_kf_T[b][0] + 8) for b in range(num_keyframes)]
  
  D_rec_T = [(D_kf_aff[-1][1] + 8*r, D_kf_aff[-1][1] + 8*r + 6) for r in range(num_rec_frames)]
  D_rec_aff = [(D_rec_T[r][0] + 6, D_rec_T[r][0] + 8) for r in range(num_rec_frames)]

  if num_rec_frames > 0:
    D_kf_s = [(D_rec_aff[-1][1] + b, D_rec_aff[-1][1] + (b+1)) for b in range(num_keyframes)]
  else:
    D_kf_s = [(D_kf_aff[-1][1] + b, D_kf_aff[-1][1] + (b+1)) for b in range(num_keyframes)]
  D_kf_d = [(D_kf_s[-1][1] + depth_inds[b].item(), D_kf_s[-1][1] + depth_inds[b+1].item()) for b in range(num_keyframes)]

  return D, D_kf_T, D_kf_aff, D_rec_T, D_rec_aff, D_kf_s, D_kf_d

def get_forward_edges(B, D_kf_s, D_kf_d):
  # Get forward consecutive keyframe edges
  ref_ids = [b for b in range(0,B-1)]
  target_ids = [b for b in range(1,B)]

  ref_pose_inds = [(8*b, 8*b + 8) for b in range(0,B-1)]
  target_pose_inds = [(8*b, 8*b + 8) for b in range(1,B)]
  ref_scale_inds = D_kf_s[0:B-1]
  target_scale_inds = D_kf_s[1:B]
  ref_depth_inds =  D_kf_d[0:B-1]
  target_depth_inds = D_kf_d[1:B]
  return ref_ids, target_ids, ref_pose_inds, target_pose_inds, \
      ref_scale_inds, target_scale_inds, ref_depth_inds, target_depth_inds

def get_backward_edges(B, D_kf_s, D_kf_d):
  ref_ids = [b for b in range(1,B)]
  target_ids = [b for b in range(0,B-1)]

  ref_pose_inds = [(8*b, 8*b + 8) for b in range(1,B)]
  target_pose_inds = [(8*b, 8*b + 8) for b in range(0,B-1)]
  ref_scale_inds = D_kf_s[1:B]
  target_scale_inds = D_kf_s[0:B-1]
  ref_depth_inds = D_kf_d[1:B]
  target_depth_inds = D_kf_d[0:B-1]
  return ref_ids, target_ids, ref_pose_inds, target_pose_inds, \
      ref_scale_inds, target_scale_inds, ref_depth_inds, target_depth_inds

def get_gradient(J, r):
  grad = -torch.sum(J * r[...,None], dim=(1,2))
  return grad

def get_hessian_diag_block(J):
  H_block = torch.einsum('bnck,bncl->bkl', J, J)
  return H_block

def get_hessian_off_diag_block(J1, J2):
  H12_blocks = torch.einsum('bnck,bncl->bkl', J1, J2)
  return H12_blocks

def accumulate_gradient(grad_block, inds, grad):
  grad[inds[0]:inds[1]] += grad_block

def accumulate_hessian_diag(H_block, inds, H):
  H[inds[0]:inds[1], inds[0]:inds[1]] += H_block

def accumulate_hessian_off_diag(H_block, inds1, inds2, H):
  H[inds1[0]:inds1[1], inds2[0]:inds2[1]] += H_block
  H[inds2[0]:inds2[1], inds1[0]:inds1[1]] += torch.transpose(H_block, dim0=0, dim1=1)

def solve_system(H, g):
  # # Check SVD of Hessian for debugging 
  # U, S, Vh = torch.linalg.svd(H)
  # print(S.flatten())
  # print(Vh[-1:,:])

  # L = torch.linalg.cholesky(H, upper=False)
  L, _ = torch.linalg.cholesky_ex(H, upper=False, check_errors=False) 
  delta = torch.cholesky_solve(g[:,None], L, upper=False)
  return delta

# poses (B,4,4)
# mean_log_depth (B-1, 1, 1)
# log_depth_ref (B-1, N_train, 1)
def update_vars(delta, poses, mean_log_depth, sparse_log_depth, aff, poses_recent, aff_recent, depth_inds):
  B = poses.shape[0]
  R = poses_recent.shape[0]
  D, D_kf_T, D_kf_aff, D_rec_T, D_rec_aff, D_kf_s, D_kf_d = get_dims(B, R, depth_inds)

  delta_T_aff = torch.reshape(delta[D_kf_T[0][0] : D_kf_aff[-1][1]], (-1,8))
  delta_T = delta_T_aff[:,:6]
  delta_ab = delta_T_aff[:,6:]

  # Update poses
  poses_new = lie.batch_se3(poses, delta_T)

  # Update affine parameters
  aff_new = aff + delta_ab[...,None]

  # Update scales
  delta_s = delta[D_kf_s[0][0] : D_kf_s[-1][1]]
  mean_log_depth_new = mean_log_depth + delta_s[...,None]

  # Update depths
  sparse_log_depth_new = torch.zeros_like(sparse_log_depth)
  for b in range(B):
    d = D_kf_d[b][1] - D_kf_d[b][0]
    sparse_log_depth_new[b,:d,:] = sparse_log_depth[b,:d,:] + delta[D_kf_d[b][0]:D_kf_d[b][1]]

  # Update recent variables
  if R > 0:
    delta_T_aff_recent = torch.reshape(delta[D_rec_T[0][0] : D_rec_aff[-1][1]], (-1,8))
    delta_T_recent = delta_T_aff_recent[:,:6]
    delta_ab_recent = delta_T_aff_recent[:,6:]
    poses_recent_new = lie.batch_se3(poses_recent, delta_T_recent)
    aff_recent_new = aff_recent + delta_ab_recent[...,None]
  else:
    poses_recent_new = poses_recent.clone()
    aff_recent_new = aff_recent.clone()

  return poses_new, mean_log_depth_new, sparse_log_depth_new, aff_new, poses_recent_new, aff_recent_new