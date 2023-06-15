import torch

from depth_cov.odom.odom_opt_utils import get_hessian_diag_block, get_hessian_off_diag_block, get_gradient, accumulate_gradient, accumulate_hessian_diag, accumulate_hessian_off_diag


def linearize_isotropic_depth_prior(depths, meas, H, g, Ddepth, sigma = 1e-6):
  device = depths.device
  N = depths.shape[0]
  info_sqrt = 1.0/sigma
  info = info_sqrt * info_sqrt
  # h(x) - z
  r = (depths-meas)[:,0]
  total_err = torch.sum(info*torch.square(r))
  # Gradient g = -Jt @ r
  gs = -r
  g[Ddepth[0]:Ddepth[1]] += info*gs
  # H = Jt @ J
  Hs = torch.eye(N, device=device)
  H[Ddepth[0]:Ddepth[1],Ddepth[0]:Ddepth[1]] += info*Hs

  return total_err

def linearize_sparse_depth_prior(L_mm):
  B, N_train, _ = L_mm.shape
  device = L_mm.device

  da_ds = -torch.ones(B, N_train, 1, device=device)
  dr_ds = torch.linalg.solve_triangular(L_mm, da_ds, upper=False).unsqueeze(2)
  da_dd = torch.eye(N_train, device=device).reshape((1,N_train,N_train)).repeat(B,1,1)
  dr_dd = torch.linalg.solve_triangular(L_mm, da_dd, upper=False).unsqueeze(2)

  H_s_s = get_hessian_diag_block(dr_ds)
  H_s_d = get_hessian_off_diag_block(dr_ds, dr_dd)
  H_d_d = get_hessian_diag_block(dr_dd)

  return dr_ds, dr_dd, H_s_s, H_s_d, H_d_d

def construct_sparse_depth_prior_system(sparse_log_depth_ref, mean_log_depth, H, g, Ds, Dd, dr_ds, dr_dd, H_s_s, H_s_d, H_d_d):
  B, N_train, _ = sparse_log_depth_ref.shape

  # h(x) - z with z = 0
  a = sparse_log_depth_ref - mean_log_depth
  r = torch.matmul(dr_dd[:,:,0,:], a)
  total_err = torch.sum(torch.square(r))

  # Gradient g = -Jt @ r
  grad_batch = get_gradient(dr_ds, r)
  accumulate_gradient(grad_batch[0,:], Ds, g)
  grad_batch = get_gradient(dr_dd, r)
  d = Dd[1] - Dd[0]
  accumulate_gradient(grad_batch[0,:d], Dd, g)
  
  accumulate_hessian_diag(H_s_s[0,:,:], Ds, H)
  accumulate_hessian_diag(H_d_d[0,:d,:d], Dd, H)
  # Off diagonal blocks
  accumulate_hessian_off_diag(H_s_d[0,:,:d], Ds, Dd, H)

  return total_err

def construct_sparse_depth_prior_system_batch(sparse_log_depth_ref, mean_log_depth, H, g, Ds, Dd, dr_ds, dr_dd, H_s_s, H_s_d, H_d_d):
  B, N_train, _ = sparse_log_depth_ref.shape

  # h(x) - z with z = 0
  a = sparse_log_depth_ref - mean_log_depth
  r = torch.matmul(dr_dd[:,:,0,:], a)

  # Scale
  grad_batch = get_gradient(dr_ds, r)
  for b in range(B):
    accumulate_gradient(grad_batch[b,:], Ds[b], g)
  for b in range(B):
    accumulate_hessian_diag(H_s_s[b,:,:], Ds[b], H)

  # Depth
  grad_batch = get_gradient(dr_dd, r)
  for b in range(B):
    d = Dd[b][1] - Dd[b][0]
    accumulate_gradient(grad_batch[b,:d], Dd[b], g)
  for b in range(B):
    d = Dd[b][1] - Dd[b][0]
    accumulate_hessian_diag(H_d_d[b,:d,:d], Dd[b], H)

  # Scale-depth
  for b in range(B):
    d = Dd[b][1] - Dd[b][0]
    accumulate_hessian_off_diag(H_s_d[b,:,:d], Ds[b], Dd[b], H)

  total_err = torch.sum(torch.square(r))

  return total_err


def linearize_mean_depth_prior_system(Knm_Kmminv, sigma):
  B, N, M = Knm_Kmminv.shape

  info_sqrt = 1.0/sigma

  dr_ds = -torch.sum(Knm_Kmminv, dim=(1,2), keepdim=True)/N
  dr_dd = torch.sum(Knm_Kmminv, dim=(1), keepdim=True)/N

  dr_ds *= info_sqrt
  dr_dd *= info_sqrt

  H_s_s = torch.einsum('hjk,hjl->hkl', dr_ds, dr_ds)
  H_s_d = torch.einsum('hjk,hjl->hkl', dr_ds, dr_dd)
  H_d_d = torch.einsum('hjk,hjl->hkl', dr_dd, dr_dd)

  return dr_ds, dr_dd, H_s_s, H_s_d, H_d_d


def construct_mean_depth_prior_system(sparse_log_depth_ref, mean_log_depth, Knm_Kmminv, H, g, Ds, Dd, dr_ds, dr_dd, H_s_s, H_s_d, H_d_d, sigma):
  B, N_train, _ = sparse_log_depth_ref.shape

  info_sqrt = 1.0/sigma

  # h(x) - z with z = 0
  r = torch.mean(torch.matmul(Knm_Kmminv, sparse_log_depth_ref - mean_log_depth), dim=(1,2), keepdim=True)
  r *= info_sqrt
  total_err = torch.sum(torch.square(r))  

  # Gradient g = -Jt @ r
  g[Ds[0]:Ds[1]] -= torch.sum(dr_ds * r, dim=(1)).flatten()
  g[Dd[0]:Dd[1]] -= torch.sum(dr_dd * r, dim=(1)).flatten()
  
  # H = Jt @ J
  # Diagonal blocks
  H[Ds[0]:Ds[1], Ds[0]:Ds[1]] += torch.block_diag(*H_s_s)
  H[Dd[0]:Dd[1], Dd[0]:Dd[1]] += torch.block_diag(*H_d_d)
  # Off diagonal blocks
  H_s_d_block = torch.block_diag(*H_s_d)
  H[Ds[0]:Ds[1], Dd[0]:Dd[1]] += H_s_d_block
  H[Dd[0]:Dd[1], Ds[0]:Ds[1]] += torch.transpose(H_s_d_block, dim0=0, dim1=1)

  return total_err