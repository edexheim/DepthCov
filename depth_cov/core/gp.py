import torch
import torch.nn as nn
import numpy as np

class GpTrainModule(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, K_train_train, y_train, mean, var):
    
    # Center data using mean
    y_centered = y_train - mean

    A = K_train_train + torch.diag_embed(var) # (B, train_points, train_points)
    L, info = torch.linalg.cholesky_ex(A, upper=False, check_errors=False) # (B, train_points, train_points)
    alpha = torch.cholesky_solve(y_centered, L, upper=False)

    # Marginal likelihood
    n = L.shape[-1]
    data_fit = torch.sum(y_centered*alpha, dim=1) # Shape of (batch_size, output_dim)
    data_term = 0.5 * torch.sum(data_fit, dim=1) # Shape of (batch size)
    complexity_term = torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=1)
    constant_term = 0.5*n*np.log(2*np.pi)
    neg_log_marg_likelihood = data_term + complexity_term + constant_term

    return L, alpha, neg_log_marg_likelihood, info


class GpTestModule(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, L, alpha, K_train_test, K_test_test_diag, mean, var=None):

    pred_mean = mean + torch.sum(alpha.unsqueeze(-2) * K_train_test.unsqueeze(-1), dim=1)

    v = torch.cholesky_solve(K_train_test, L, upper=False)
    pred_var = K_test_test_diag - torch.sum(K_train_test * v, dim=1)
    pred_var = pred_var.unsqueeze(-1)

    if var is not None:
      pred_var += var

    return pred_mean, pred_var

class GpTestFullCovModule(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, L, alpha, K_train_test, K_test_test, mean, var=None):

    pred_mean = mean + torch.sum(alpha.unsqueeze(-2) * K_train_test.unsqueeze(-1), dim=1)

    v = torch.cholesky_solve(K_train_test, L, upper=False)
    tmp = torch.matmul(torch.transpose(K_train_test, 1, 2), v)
    pred_cov = K_test_test - tmp
    if var is not None:
      pred_cov += torch.diag(var)
    pred_cov = pred_cov.unsqueeze(-1)

    return pred_mean, pred_cov