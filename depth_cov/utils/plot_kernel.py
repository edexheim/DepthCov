import torch
import numpy as np


def plot_multivariate_gaussian(mu, Sigma, sigma_range=3.0):
  device = mu.device

  e,Q = torch.linalg.eigh(Sigma)

  Sigma_sqrt = Q @ torch.diag(torch.sqrt(e)) @ Q.T

  N = 1001
  sigma_range = torch.linspace(-sigma_range, sigma_range, N, device=device)
  x, y = torch.meshgrid(sigma_range, sigma_range, indexing='ij')

  pos = torch.empty(x.shape + (2,), device=device)
  pos[:, :, 0] = x
  pos[:, :, 1] = y
  
  transformed_pos = torch.squeeze(Sigma_sqrt[None,None,:] @ pos[:,:,:,None])
  transformed_pos[:,:,None] += mu

  n = mu.shape[0]
  Sigma_det = torch.linalg.det(Sigma)
  Sigma_inv = torch.linalg.inv(Sigma) 

  norm_constant = torch.sqrt((2*np.pi)**n * Sigma_det)
  fac = torch.einsum('...k,kl,...l->...', transformed_pos-mu, Sigma_inv, transformed_pos-mu)

  x_t = transformed_pos[:,:,0]
  y_t = transformed_pos[:,:,1]
  z = torch.exp(-fac / 2) / norm_constant

  max_border = max(torch.max(z[N-1,:]), torch.max(z[:,N-1]))
  z = np.ma.array(z, mask=z < max_border)

  return x_t, y_t, z