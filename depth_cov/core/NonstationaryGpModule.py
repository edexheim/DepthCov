from random import gauss
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
import numpy as np

from depth_cov.core.covariance import CovarianceModule, CrossCovarianceModule, DiagonalCovarianceModule
import depth_cov.core.gaussian_kernel as gk
from depth_cov.core.gp import GpTrainModule, GpTestModule
from depth_cov.core.gp_vfe import GpVfeModuleConstantNoise
from depth_cov.core.kernels import squared_exponential, matern
from depth_cov.core.samplers import sample_sparse_coords_norm
from depth_cov.data.depth_resize import resize_depth
from depth_cov.nn.UNet import UNet
from depth_cov.utils.utils import normalize_coordinates, unnormalize_coordinates, downsample_depth, get_test_coords, bilinear_interpolation, sample_coords

class NonstationaryGpModule(pl.LightningModule):

  def __init__(self):
    super().__init__()

    self.automatic_optimization = False

    self.max_train_samples = 256
    self.max_test_samples = 256*192//2 
    num_levels = 5
    self.depth_var_prior = 1e-2
    kernel_scale_prior = 1e0

    self.gaussian_cov_net = UNet(num_levels=num_levels, in_channels=3,
              base_feature_channels=16, feature_channels=3,
              kernel_size=3,
              padding=1,
              stride=1,
              feature_act = gk.normalize_params_cov)

    # Covariance modules and parameters
    iso_cov_fn = matern

    self.log_depth_var_priors = []  
    self.log_depth_var_scales = nn.ParameterList()
    self.cov_modules = nn.ModuleList()
    self.cross_cov_modules = nn.ModuleList()
    self.diagonal_cov_modules = nn.ModuleList()
    
    for i in range(num_levels-1):
      self.log_depth_var_priors.append(self.depth_var_prior)
      var_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)
      self.log_depth_var_scales.append(var_param)

      kernel_scale = nn.Parameter(torch.tensor(0.0), requires_grad=True)
      self.cov_modules.append(CovarianceModule(iso_cov_fn=iso_cov_fn, scale_param=kernel_scale, scale_prior=kernel_scale_prior))
      self.cross_cov_modules.append(CrossCovarianceModule(iso_cov_fn=iso_cov_fn, scale_param=kernel_scale, scale_prior=kernel_scale_prior))
      self.diagonal_cov_modules.append(DiagonalCovarianceModule(iso_cov_fn=iso_cov_fn, scale_param=kernel_scale, scale_prior=kernel_scale_prior))

    # GP Modules
    self.gp_train_module = GpTrainModule()
    self.gp_test_module = GpTestModule()
    self.gp_vfe_module = GpVfeModuleConstantNoise()


  def get_var(self, level):
    var_level = self.log_depth_var_priors[level] * torch.exp(self.log_depth_var_scales[level])
    return var_level

  def get_scale(self, level):
    return self.cov_modules[level].get_scale()

  def forward(self, rgb):
    gaussian_cov_params = self.gaussian_cov_net(rgb)
    num_levels = len(gaussian_cov_params)
    gaussian_covs = []
    for l in range(0, num_levels):
      gaussian_covs.append(gk.kernel_params_to_covariance(gaussian_cov_params[l]))

    return gaussian_covs


  def condition_level(self, gaussian_covs, level, sparse_coords_norm, sparse_depth, mean_depth, test_size):
    device = gaussian_covs[level].device
    b = gaussian_covs[level].shape[0]
    
    # Unnormalized coords must be in train image reference frame!
    test_coords = get_test_coords(test_size, device, batch_size=b)
    test_coords_norm = normalize_coordinates(test_coords, test_size)

    sparse_var_level = self.get_var(level)
    sparse_vars = sparse_var_level*torch.ones_like(sparse_depth)
    sparse_vars = sparse_vars.squeeze(-1)

    E_train = gk.interpolate_kernel_params(gaussian_covs[level], sparse_coords_norm)
    E_test = gk.interpolate_kernel_params(gaussian_covs[level], test_coords_norm)

    K_train_train = self.cov_modules[level](sparse_coords_norm, E_train)
    K_train_test = self.cross_cov_modules[level](sparse_coords_norm, E_train, test_coords_norm, E_test)
    K_test_test_diag = self.diagonal_cov_modules[level](test_coords_norm, E_test)

    L, alpha, _, info = self.gp_train_module(K_train_train, sparse_depth, mean_depth, sparse_vars)
    if info.any():
      print("Cholesky failed")
    pred_depth, pred_var = self.gp_test_module(L, alpha, K_train_test, K_test_test_diag, mean_depth)
    pred_depth = torch.permute(pred_depth, (0,2,1))
    pred_depth = torch.reshape(pred_depth, (b,1,test_size[0],test_size[1]))
    pred_var = torch.permute(pred_var, (0,2,1))
    pred_var = torch.reshape(pred_var, (b,1,test_size[0],test_size[1]))

    return pred_depth, pred_var, L, E_train

  def condition(self, gaussian_covs, sparse_coords_norm, sparse_log_depth, mean_depth, test_size):
    num_levels = len(gaussian_covs)
    pred_log_depths = []
    pred_vars = []
    for l in range(num_levels):
      pred_log_depth, pred_var, _, _ = self.condition_level(gaussian_covs, l, 
          sparse_coords_norm, sparse_log_depth, mean_depth, test_size)
      pred_log_depths.append(pred_log_depth)
      pred_vars.append(pred_var)

      if ( (pred_var<=0.0).any() ):
        print("Found non-positive variance", torch.min(pred_var))

    return pred_log_depths, pred_vars

  def get_covariance(self, gaussian_covs, level, coords_norm, E=None):
    if E is None:
      E = gk.interpolate_kernel_params(gaussian_covs[level], coords_norm)
    K = self.cov_modules[level](coords_norm, E)
    return K, E

  def get_covariance_with_noise(self, gaussian_covs, level, coords_norm, E=None):
    K, E = self.get_covariance(gaussian_covs, level, coords_norm, E=E)
    b, m, _ = K.shape
    noise = self.get_var(level) * torch.ones(b,m,device=K.device)
    K += torch.diag_embed(noise)
    return K, E

  def get_covariance_chol(self, gaussian_covs, level, coords_norm):
    K, E = self.get_covariance_with_noise(gaussian_covs, level, coords_norm)
    b, m, _ = K.shape
    device = K.device

    L, info = torch.linalg.cholesky_ex(K, upper=False)
    return L, E

  def get_cross_covariance(self, gaussian_covs, level, coords1_norm, coords2_norm, E1=None, E2=None):
    if E1 is None:
      E1 = gk.interpolate_kernel_params(gaussian_covs[level], coords1_norm)
    if E2 is None:
      E2 = gk.interpolate_kernel_params(gaussian_covs[level], coords2_norm)

    K_mn = self.cross_cov_modules[level](coords1_norm, E1, coords2_norm, E2)
    return K_mn

  def get_diagonal_covariance(self, gaussian_covs, level, coords_norm, E=None):
    if E is None:
      E = gk.interpolate_kernel_params(gaussian_covs[level], coords_norm)
    K_diag = self.diagonal_cov_modules[level](coords_norm, E)
    return K_diag
  
  def get_correlation_map(self, gaussian_covs, level, coords_m_norm, test_size):
    b, _, h, w = gaussian_covs[-1].shape
    device = gaussian_covs[-1].device 
    coords_n = get_test_coords(test_size, device, batch_size=b)
    coords_n_norm = normalize_coordinates(coords_n, test_size)

    E_m = gk.interpolate_kernel_params(gaussian_covs[level], coords_m_norm)
    E_n = gk.interpolate_kernel_params(gaussian_covs[level], coords_n_norm)

    K_mn = self.cross_cov_modules[level](coords_m_norm, E_m, coords_n_norm, E_n)
    K_m_map = torch.reshape(K_mn, (b, h, w))
    return K_m_map

  def get_linear_predictor(self, gaussian_covs, level, coords_m_norm, coords_n_norm):
    L_mm, E_m = self.get_covariance_chol(gaussian_covs, level, coords_m_norm)
    E_n = gk.interpolate_kernel_params(gaussian_covs[level], coords_n_norm)

    K_mn = self.cross_cov_modules[level](coords_m_norm, E_m, coords_n_norm, E_n)
    Kmminv_Kmn = torch.cholesky_solve(K_mn, L_mm, upper=False)
    Knm_Kmminv = torch.transpose(Kmminv_Kmn, dim0=-2, dim1=-1)
    
    return Knm_Kmminv, L_mm, E_m

  @staticmethod
  def get_chol_features(K_mn, L):
    nystrom = torch.linalg.solve_triangular(L, K_mn, upper=False)
    return nystrom
  
  @staticmethod
  def get_nystrom_features(K_mn, K_mm):
    s, Q = torch.linalg.eigh(K_mm)
    D_inv_sqrt = torch.diag_embed(1.0/torch.sqrt(s))
    nystrom = torch.matmul(torch.transpose(Q, dim0=-2, dim1=-1), K_mn)
    nystrom = torch.matmul(D_inv_sqrt, nystrom)    
    return nystrom

  @staticmethod
  def solve_compact_depth_hierarchical(L_mm, Knm_Kmminv, mean_log_depth, log_depth_obs, log_depth_stdev_inv):
    batch_size, n, m = Knm_Kmminv.shape
    device = Knm_Kmminv.device

    A = torch.empty((batch_size, m+n, m), device=device)
    identity = torch.eye(m, device=device).reshape((1,m,m)).repeat(batch_size,1,1)
    L_inv = torch.linalg.solve_triangular(L_mm, identity, upper=False)
    A[:,:m,:] = L_inv
    A[:,m:,:] = log_depth_stdev_inv*Knm_Kmminv

    b = torch.empty((batch_size, m+n, 1), device=device)
    b[:,:m,:] = torch.linalg.solve_triangular(L_mm, mean_log_depth*torch.ones((batch_size, m, 1), device=device), upper=False)
    b[:,m:,:] = log_depth_stdev_inv*(log_depth_obs + torch.sum(Knm_Kmminv * mean_log_depth, dim=(2), keepdim=True) - mean_log_depth)

    compact_log_depth, _, _, _ = torch.linalg.lstsq(A, b)

    residuals = torch.matmul(A, compact_log_depth) - b
    total_err = torch.sum(torch.square(residuals), dim=(1,2))

    return compact_log_depth, total_err

  @staticmethod
  def solve_compact_depth(Knm_Kmminv, log_depth_obs, mean_log_depth):
    A = Knm_Kmminv
    b = log_depth_obs + torch.sum(Knm_Kmminv * mean_log_depth, dim=(2), keepdim=True) - mean_log_depth
    compact_log_depth, _, _, _ = torch.linalg.lstsq(A, b)

    residuals = torch.matmul(A, compact_log_depth) - b
    total_err = torch.sum(torch.square(residuals), dim=(1,2))
    
    return compact_log_depth, total_err

  @staticmethod
  def solve_mean_depth(Knm_Kmminv, log_depth_obs, sparse_log_depth):
    A = 1.0 - torch.sum(Knm_Kmminv, dim=(2), keepdim=True)
    b = log_depth_obs - torch.matmul(Knm_Kmminv, sparse_log_depth)

    mean_log_depth, _, _, _ = torch.linalg.lstsq(A, b)

    residuals = torch.matmul(A, mean_log_depth) - b
    total_err = torch.sum(torch.square(residuals), dim=(1,2))

    return mean_log_depth, total_err

  @staticmethod
  def solve_compact_and_mean_depth(Knm_Kmminv, log_depth_obs):
    b, n, m = Knm_Kmminv.shape
    device = Knm_Kmminv.device

    A = torch.empty((b,n,m+1), device=device)
    A[:,:,:m] = Knm_Kmminv
    A[:,:,m:] = 1.0 - torch.sum(Knm_Kmminv, dim=(2), keepdim=True)
    x, _, _, _ = torch.linalg.lstsq(A, log_depth_obs)
    compact_log_depth = x[:,:m,:]
    mean_log_depth = x[:,m:,:]

    residuals = torch.matmul(A, x) - log_depth_obs
    total_err = torch.sum(torch.square(residuals), dim=(1,2))

    return compact_log_depth, mean_log_depth, total_err

  @staticmethod
  def woodbury(K_mm, K_mn, var):
    b, m, n = K_mn.shape
    var_inv = 1.0/var

    A = var * K_mm + torch.matmul(K_mn, torch.transpose(K_mn, dim0=-2, dim1=-1))
    L_A, info_A = torch.linalg.cholesky_ex(A, upper=False)
    K_nn_approx = torch.matmul(
        torch.transpose(K_mn, dim0=-2, dim1=-1), 
        torch.cholesky_solve(K_mn, L_A))
    K_nn_approx += torch.diag_embed(torch.ones(b,n,device=K_mm.device))
    K_nn_approx *= var_inv
    return 

  def prep_train_level(self, level, num_levels, gt_depth, gaussian_covs):
    b = gt_depth.shape[0]
    device = gt_depth.device

    domain_level = level

    with torch.no_grad():
      # Downsample depth
      depth = resize_depth(gt_depth, mode="bilinear", size=gaussian_covs[level].shape[-2:])
      # Sample coords
      coord_train, depth_train, batch_train = sample_coords(depth, gaussian_covs, self.max_train_samples, mode="uniform")
      if coord_train is None:
        return None, None, None, None
      batch_train = torch.arange(b,device=device).unsqueeze(1).repeat(1,coord_train.shape[1])
      coord_test, depth_test, batch_test = sample_coords(depth, None, self.max_test_samples, mode="uniform")
      if coord_test is None:
        return None, None, None, None
  
    E_train = gaussian_covs[level][batch_train, :, coord_train[...,0], coord_train[...,1]]
    E_train = torch.reshape(E_train, (E_train.shape[0], E_train.shape[1], 2, 2))
    E_test = gaussian_covs[level][batch_test, :, coord_test[...,0], coord_test[...,1]]
    E_test = torch.reshape(E_test, (E_test.shape[0], E_test.shape[1], 2, 2))

    with torch.no_grad():
      coord_train_norm = normalize_coordinates(coord_train, gaussian_covs[level].shape[-2:])
      coord_test_norm = normalize_coordinates(coord_test, gaussian_covs[level].shape[-2:])

    K_train_train = self.cov_modules[level](coord_train_norm, E_train)
    K_train_test = self.cross_cov_modules[level](coord_train_norm, E_train, coord_test_norm, E_test)
    K_test_test_diag = self.diagonal_cov_modules[level](coord_test_norm, E_test)

    return K_train_train, K_train_test, K_test_test_diag, depth_test

  def get_loss(self, batch, mode):
    rgb, gt_depth = batch
    B = rgb.shape[0]
    device = rgb.device

    # Network
    gaussian_cov_params = self.gaussian_cov_net(rgb)
    num_levels = len(gaussian_cov_params)
    gaussian_covs = []
    for l in range(0, num_levels):
      gaussian_covs.append(gk.kernel_params_to_covariance(gaussian_cov_params[l]))


    num_valid_levels = 0
    neg_log_marg_likelihood = torch.zeros((B), device=device)
    num_pixels = []
    for l in range(num_levels):
      sparse_var_level = self.get_var(l)
      K_mm, K_mn, K_nn_diag, depth_test = self.prep_train_level(l, num_levels, gt_depth, gaussian_covs)
      if K_mm is None:
        continue

      num_pixels.append(depth_test.shape[1])

      mean_depth = torch.mean(depth_test, [1,2], keepdim=True)
      neg_log_marg_likelihood_level, info = self.gp_vfe_module(K_mm.double(), K_mn.double(), K_nn_diag.double(), depth_test.double(), mean_depth, sparse_var_level)
      neg_log_marg_likelihood_level = neg_log_marg_likelihood_level.float()
      # Account for higher pixel count (since NLML is calculated as a mean for stability)
      level_scale_factor = num_pixels[-1]/num_pixels[0]
      neg_log_marg_likelihood_level *= level_scale_factor

      params_extrema = torch.tensor([
          torch.min(gaussian_cov_params[l][:,0,:,:]).item(), torch.max(gaussian_cov_params[l][:,0,:,:]).item(), torch.median(gaussian_cov_params[l][:,0,:,:]).item(),\
          torch.min(gaussian_cov_params[l][:,1,:,:]).item(), torch.max(gaussian_cov_params[l][:,1,:,:]).item(), torch.median(gaussian_cov_params[l][:,1,:,:]).item(),\
          torch.min(gaussian_cov_params[l][:,2,:,:]).item(), torch.max(gaussian_cov_params[l][:,2,:,:]).item(), torch.median(gaussian_cov_params[l][:,2,:,:]).item() \
        ])
      if info.any():
        print("Cholesky failed " + str(l))
        print(params_extrema)
        continue
      elif neg_log_marg_likelihood_level.isnan().any():
        print("NaN NLML found in level " + str(l))
        print(params_extrema)
        continue
      else:
        num_valid_levels += 1

      neg_log_marg_likelihood += neg_log_marg_likelihood_level


    with torch.no_grad():
      params_extrema = torch.tensor([
          torch.min(gaussian_cov_params[-1][:,0,:,:]).item(), torch.max(gaussian_cov_params[-1][:,0,:,:]).item(), torch.mean(gaussian_cov_params[-1][:,0,:,:]).item(),\
          torch.min(gaussian_cov_params[-1][:,1,:,:]).item(), torch.max(gaussian_cov_params[-1][:,1,:,:]).item(), torch.mean(gaussian_cov_params[-1][:,1,:,:]).item(),\
          torch.min(gaussian_cov_params[-1][:,2,:,:]).item(), torch.max(gaussian_cov_params[-1][:,2,:,:]).item(), torch.mean(gaussian_cov_params[-1][:,2,:,:]).item() \
      ])

    # Training set loss
    if num_valid_levels == 0:
      return None
    if neg_log_marg_likelihood.isnan().any():
      print("NLL found nan")
      print(params_extrema)
      return None
    neg_log_marg_likelihood_mean = torch.mean(neg_log_marg_likelihood)
    if mode == "train":
      self.manual_backward(neg_log_marg_likelihood_mean)
    final_level_train_nll = neg_log_marg_likelihood_mean.detach()

    return {"final_level_nll": final_level_train_nll, "params_extrema": params_extrema}

  def training_step(self, batch, batch_idx):
    if batch is None:
      return None

    opt = self.optimizers()
    outputs = self.get_loss(batch, mode="train")
    if outputs is None:
      return None

    opt.step()
    opt.zero_grad()

    return outputs

  def training_epoch_end(self, outputs):
    if len(outputs) > 0:
      loss = torch.mean(torch.stack([x['final_level_nll'] for x in outputs]))
      self.log("loss_train", loss, on_epoch=True)

      params_extrema = torch.stack([x['params_extrema'] for x in outputs])
      min_params = torch.min(params_extrema[:,0::3], dim=0).values
      max_params = torch.max(params_extrema[:,1::3], dim=0).values
      median_params = torch.median(params_extrema[:,2::3], dim=0).values

      self.log("scale_min", min_params[0], on_epoch=True)
      self.log("theta_min", min_params[1], on_epoch=True)
      self.log("elongation_min", min_params[2], on_epoch=True)
      self.log("scale_max", max_params[0], on_epoch=True)
      self.log("theta_max", max_params[1], on_epoch=True)
      self.log("elongation_max", max_params[2], on_epoch=True)
      self.log("scale_median", median_params[0], on_epoch=True)
      self.log("theta_median", median_params[1], on_epoch=True)
      self.log("elongation_median", median_params[2], on_epoch=True)

      self.log("kernel_scale", self.cov_modules[-1].get_scale(), on_epoch=True)
      sparse_var_level = self.get_var(-1)
      self.log("depth_var", sparse_var_level.item(), on_epoch=True)

    return


  def validation_step(self, batch, batch_idx):
    if batch is None:
      return None
    loss = self.get_loss(batch, mode="val")
    return loss   

  def validation_epoch_end(self, outputs):
    if len(outputs) > 0:
      loss = torch.mean(torch.stack([x['final_level_nll'] for x in outputs]))
      self.log("loss_val", loss, on_epoch=True)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
    return [optimizer]