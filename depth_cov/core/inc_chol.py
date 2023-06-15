import torch

# Assumess L has space allocated and is updated in-place
def update_chol_inplace(L, k_ni, k_ii, ind : int):
  l_ni2 = torch.linalg.solve_triangular(L, k_ni, upper=False)
  L[:,ind:ind+1,:] = torch.transpose(l_ni2, dim0=1, dim1=2)
  L[:,ind:ind+1,ind:ind+1] = torch.sqrt(k_ii - torch.sum(torch.square(l_ni2), dim=1, keepdim=True))

# L_inv @ Kmn
def update_obs_info_inplace(k_id, L, obs_info, ind : int):
  l_ni = torch.transpose(L[:,ind:ind+1,:ind], dim0=1, dim1=2)
  obs_info[:,ind:ind+1,:] = (k_id - torch.sum(l_ni * obs_info[:,:ind,:], dim=1, keepdim=True))/L[:,ind:ind+1,ind:ind+1]

def get_new_chol_obs_info(L, obs_info, k_ni, k_id, k_ii):
  l_ni = torch.linalg.solve_triangular(L, k_ni, upper=False)
  l_ii = torch.sqrt(k_ii - torch.sum(torch.square(l_ni), dim=1, keepdim=True))
  obs_info_new = torch.div(k_id - torch.sum(l_ni * obs_info, dim=1, keepdim=True), l_ii)
  return l_ni, l_ii, obs_info_new