import torch
from torch.utils.data import WeightedRandomSampler

import depth_cov.core.gaussian_kernel as gk
from depth_cov.core.gp import GpTrainModule, GpTestModule
from depth_cov.core.inc_chol import update_chol_inplace, update_obs_info_inplace, get_new_chol_obs_info
from depth_cov.utils.lin_alg import det2x2, trace2x2
from depth_cov.utils.utils import normalize_coordinates, unnormalize_coordinates

import depth_cov_backends

import time

def get_domain(gaussian_covs, mask=None, border=0):
  b, c, h, w = gaussian_covs.shape
  device = gaussian_covs.device

  y_coords, x_coords = torch.meshgrid(torch.arange(h, dtype=torch.long, device=device), torch.arange(w, dtype=torch.long, device=device), indexing='ij')
  coord_img = torch.dstack((y_coords, x_coords))
  coord_img = coord_img[border:h-border,border:w-border,:]
  if mask is None:
    mask = torch.ones(coord_img.shape[:2], device=device, dtype=torch.bool)
  else:
    mask = mask[border:h-border,border:w-border]
  coord_vec = coord_img[mask,:]
  coord_vec = coord_vec.unsqueeze(0).repeat(b,1,1)

  gaussian_covs_vec = gaussian_covs[:,:,coord_vec[0,:,0],coord_vec[0,:,1]]
  gaussian_cov_mats = torch.permute(gaussian_covs_vec, (0,2,1))
  gaussian_cov_mats = torch.reshape(gaussian_cov_mats, (b,-1,2,2)).contiguous()

  return coord_vec, gaussian_cov_mats

# Returns normalized coordinates
def sample_sparse_coords_norm(gaussian_covs, num_samples, 
    mode, max_stdev_thresh = -1e8,
    curr_coords_norm=None, 
    model=None, model_level=-1,
    mask=None, dtype = torch.float):
  
  device = "cuda"

  orig_dtype = gaussian_covs[model_level].dtype
  orig_device = gaussian_covs[model_level].device

  b, _, h, w = gaussian_covs[model_level].shape
  if curr_coords_norm is None:
    curr_coords_norm = torch.empty((gaussian_covs[model_level].shape[0],0,2), device=device, dtype=dtype)

  gaussian_covs[model_level] = gaussian_covs[model_level].to(device=device, dtype=dtype)
  curr_coords_norm = curr_coords_norm.to(device=device, dtype=dtype)
  
  coords_domain, E_domain = get_domain(gaussian_covs[model_level], mask=mask)
  coords_domain_norm = normalize_coordinates(coords_domain, gaussian_covs[model_level].shape[-2:]).to(dtype)

  if mode == "random_uniform":
    num_curr_coords = curr_coords_norm.shape[-2]
    num_samples_remaining = num_samples - num_curr_coords
    coords_norm = random_uniform(num_samples_remaining, coords_domain_norm)
    coords_norm = torch.cat((curr_coords_norm, coords_norm), dim=1)
  elif mode == "greedy_conditional_entropy":
    cov_scale = model.cross_cov_modules[model_level].get_scale().to(device)
    coords_norm = greedy_conditional_entropy(
        E_domain, num_samples, coords_domain_norm, curr_coords_norm, 
        cov_scale, max_stdev_thresh)
  else:
    raise ValueError("sample_sparse_coords mode: " + mode + " is not implemented.")

  gaussian_covs[model_level] = gaussian_covs[model_level].to(device=orig_device, dtype=orig_dtype)
  coords_norm = coords_norm.to(device=orig_device, dtype=orig_dtype)

  return coords_norm

def random_uniform(n, coords_domain_norm):
  b = coords_domain_norm.shape[0]
  device = coords_domain_norm.device
  weights = torch.ones((coords_domain_norm.shape[:-1]), device=device)
  sample_inds = torch.multinomial(weights, n, replacement=False)
  batch_inds = torch.arange(b,device=device).unsqueeze(1).repeat(1,n)
  coord_samples = coords_domain_norm[batch_inds, sample_inds, :]
  return coord_samples

def get_obs_info(L, K_mn):
  obs_info = torch.linalg.solve_triangular(L, K_mn, upper=False)
  return obs_info

def calc_var(obs_info, K_diag):
  return K_diag - torch.sum(obs_info*obs_info, dim=1)

def batch_var(coords_train_norm, E_train, coords_test_norm, E_test, model, model_level):
  b, m, _ = coords_train_norm.shape
  K_mm = model.cov_modules[model_level](coords_train_norm, E_train)
  noise = model.get_var(model_level) * torch.ones(b,m,device=coords_train_norm.device)
  K_mn = model.cross_cov_modules[model_level](coords_train_norm, E_train, coords_test_norm, E_test)
  K_nn_diag = model.diagonal_cov_modules[model_level](coords_test_norm, E_test)
  L = torch.linalg.cholesky(K_mm, upper=False)

  v = torch.linalg.solve_triangular(L, K_mn, upper=False)
  pred_var = K_nn_diag - torch.sum(v * v, dim=1)

  return pred_var

def precalc_entropy_vars(E_domain, n, coords_domain_norm, curr_coords_norm, scale):
  
  b, m, _ = curr_coords_norm.shape
  device = E_domain.device
  dtype = E_domain.dtype

  domain_size = coords_domain_norm.shape[-2]

  # Incrementally updating variables
  coords_n_norm = torch.empty((b,n,2), device=device, dtype=dtype)
  E_n = torch.empty((b,n,2,2), device=device, dtype=dtype)
  # Identity allows us to solve triangular system of 0s without worrying about effect on sums (solution is 0s)
  L = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).repeat(b,1,1)
  obs_info = torch.zeros((b,n,domain_size), device=device, dtype=dtype)

  if m > 0: # Existing coords
    raise ValueError("Sampler needs update with existing coords!")
  else: # Pick valid point with largest determinant
    cov_areas = E_domain[...,0,0]*E_domain[...,1,1] - E_domain[...,0,1] * E_domain[...,1,0]
    scale_vec = torch.reshape(cov_areas, (b, -1))
    best_inds = torch.argmax(scale_vec, dim=1)
    batch_inds = torch.arange(b, device=device)
    coords_n_norm[:,0:1,:] = coords_domain_norm[batch_inds,best_inds,:].unsqueeze(1)
    E_n[:,0:1,:,:] = E_domain[batch_inds,best_inds,:,:].unsqueeze(1)
    m = 1

  K_nn = depth_cov_backends.cross_covariance(coords_n_norm[:,:m,:], E_n[:,:m,:,:], coords_n_norm[:,:m,:].clone(), E_n[:,:m,:,:].clone(), scale)
  L[:,:m,:m] = torch.linalg.cholesky(K_nn, upper=False)

  K_md_init = depth_cov_backends.cross_covariance(coords_n_norm[:,:m,:], E_n[:,:m,:,:], coords_domain_norm, E_domain, scale)
  obs_info[:,:m,:] = get_obs_info(L[:,:m,:m], K_md_init)

  return coords_n_norm, E_n, L, obs_info, m

def greedy_loop(coords_n_norm, E_n, coords_domain_norm, E_domain, L, obs_info, m, n, scale, max_stdev_thresh):
  device = coords_n_norm.device
  b = coords_n_norm.shape[0]

  max_var_thresh = max_stdev_thresh**2
  batch_inds = torch.arange(b, device=device)

  pred_var = calc_var(obs_info[:,:m,:], scale)
  max_ret = torch.max(pred_var, dim=1)
  best_ind = max_ret.indices[0]

  for i in range(m, n):
    # Check for early termination
    if max_ret.values[0] < max_var_thresh:
      coords_n_norm = coords_n_norm[:,:i,:]
      break

    # Add new point/cov
    coord_i_norm = coords_domain_norm[batch_inds,best_ind,:].unsqueeze(1)
    E_i = E_domain[batch_inds,best_ind,:,:].unsqueeze(1)
    coords_n_norm[:,i:i+1,:] = coord_i_norm
    E_n[:,i:i+1,:,:] = E_i

    k_ni = depth_cov_backends.cross_covariance(coords_n_norm[:,0:i,:], E_n[:,0:i,:,:], coord_i_norm, E_i, scale)
    k_id = depth_cov_backends.cross_covariance(coord_i_norm, E_i, coords_domain_norm, E_domain, scale)    
    k_ii = scale

    depth_cov_backends.get_new_chol_obs_info(L, obs_info, pred_var, k_ni, k_id, k_ii, i)

    max_ret = torch.max(pred_var, dim=1)
    best_ind = max_ret.indices[0]

  return coords_n_norm

def greedy_conditional_entropy(E_domain, n, coords_domain_norm, curr_coords, cov_scale, max_stdev_thresh):

  device = E_domain.device 
  b = E_domain.shape[0]
  batch_inds = torch.arange(b, device=device)

  coords_n_norm, E_n, L, obs_info, m = precalc_entropy_vars(
      E_domain, n, coords_domain_norm, curr_coords, cov_scale)

  sampled_coords_norm = greedy_loop(coords_n_norm, E_n, coords_domain_norm, E_domain, L, obs_info, m, n, cov_scale, max_stdev_thresh)

  return sampled_coords_norm
