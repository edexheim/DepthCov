import torch
import torch.nn as nn
import numpy as np
from depth_cov.utils.lin_alg import chol_log_det, trace

class GpVfeModuleConstantNoise(nn.Module):
  def __init__(self):
    super().__init__()

  def solve_optimal_mean(self, y, L_A, K_mn, K_mn_y):
    n = y.shape[1]
    
    L_inv_Kmn_y = torch.linalg.solve_triangular(L_A, K_mn_y, upper=False)
    Kmn_sum = torch.sum(K_mn, dim=(2), keepdim=True)
    L_inv_Kmn_sum = torch.linalg.solve_triangular(L_A, Kmn_sum, upper=False)

    A1T_A1 = n
    A1T_b1 = torch.sum(y, dim=(1), keepdim=True)
    tmp = torch.transpose(L_inv_Kmn_sum, dim0=1, dim1=2)
    A2T_A2 = torch.matmul(tmp, L_inv_Kmn_sum)
    A2T_b2 = torch.matmul(tmp, L_inv_Kmn_y)

    mean = (A1T_b1-A2T_b2)/(A1T_A1-A2T_A2)
    return mean

  # Assumes variance is a scalar
  def forward(self, K_mm, K_mn, K_nn_diag, y, mean, var):
    B = K_mm.shape[0]
    M = K_mn.shape[-2]
    N = K_mn.shape[-1]
    var_inv = 1.0/var

    jitter = (1e-4)*torch.ones(B,M,device=K_mm.device)
    K_mm += torch.diag_embed(jitter)
    # Cholesky for inverses
    L_mm, info_mm = torch.linalg.cholesky_ex(K_mm, upper=False)
    A = var * K_mm + torch.matmul(K_mn, torch.transpose(K_mn, dim0=-2, dim1=-1))
    L_A, info_A = torch.linalg.cholesky_ex(A, upper=False)

    info = info_mm + info_A

    with torch.no_grad():
      K_mn_y = torch.matmul(K_mn, y)
      mean2 = self.solve_optimal_mean(y, L_A, K_mn, K_mn_y)

    y_centered2 = y - mean2
    K_mn_y_centered2 = torch.matmul(K_mn, y_centered2)

    data_term2 = 0.5*var_inv \
        * (torch.mean(y_centered2*y_centered2,dim=1).squeeze() 
          - (1/N)*torch.matmul(
              torch.transpose(K_mn_y_centered2, dim0=-2, dim1=-1), 
              torch.cholesky_solve(K_mn_y_centered2, L_A)).squeeze() )
    complexity_term2 = 0.5/N * ((N-M) * torch.log(var) + chol_log_det(L_A) - chol_log_det(L_mm))
    constant_term2 = 0.5*np.log(2*np.pi)
    trace_term2 = 0.5*var_inv * (torch.mean(K_nn_diag, dim=1) - torch.sum(K_mn/N * torch.cholesky_solve(K_mn, L_mm), dim=(1,2)))

    neg_log_marg_likelihood_mean2 = data_term2 + complexity_term2 + constant_term2 + trace_term2

    return neg_log_marg_likelihood_mean2, info

class GpVfeModuleVaryingNoise(nn.Module):
  def __init__(self):
    super().__init__()
  
  # Assumes variance is a vector
  def forward(self, K_mm, K_mn, K_nn_diag, y, mean, var):
    B = K_mm.shape[0]
    M = K_mn.shape[-2]
    N = K_mn.shape[-1]
    
    var = var.squeeze(-1)
    y_centered = (y - mean).squeeze(-1)
    var_inv = 1.0/var

    jitter = (1e-4)*torch.ones(B,M,device=K_mm.device)
    K_mm += torch.diag_embed(jitter)

    C = K_mn * var_inv[:,None,:] 
    A = K_mm + torch.matmul(C, torch.transpose(K_mn, dim0=-2, dim1=-1))
    C_y = torch.matmul(C, y_centered.unsqueeze(-1))

    # Cholesky for inverses
    L_mm, info_mm = torch.linalg.cholesky_ex(K_mm, upper=False)
    L_A, info_A = torch.linalg.cholesky_ex(A, upper=False)

    info = info_mm + info_A

    # Marginal likelihood
    data_term = 0.5 * ( torch.mean(y_centered*y_centered*var_inv,dim=1).squeeze() 
          - (1/N)*torch.matmul(torch.transpose(C_y, dim0=-2, dim1=-1), torch.cholesky_solve(C_y, L_A)).squeeze() )
    complexity_term = 0.5/N *( torch.sum(torch.log(var), dim=1) + chol_log_det(L_A) - chol_log_det(L_mm) )
    constant_term = 0.5*np.log(2*np.pi)
    trace_term = 0.5 * ( torch.mean(var_inv * K_nn_diag, dim=1) 
          - torch.sum(C/N * torch.cholesky_solve(K_mn, L_mm), dim=(1,2)) )

    neg_log_marg_likelihood = data_term + complexity_term + constant_term + trace_term

    return neg_log_marg_likelihood, info