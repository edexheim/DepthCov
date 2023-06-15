import torch

def squared_error(r):
  w = torch.ones_like(r)
  # rho = r*r
  return w # rho

def huber(r):
  k = 1.345
  
  unit = torch.ones((1), dtype=r.dtype, device=r.device)

  r_abs = torch.abs(r)
  mask = r_abs < k
  w = torch.where(mask, unit, k/r_abs)
  # rho = torch.where(mask, 0.5*r*r, k*(r_abs-0.5*k))
  return w # rho

def tukey(r, t=4.6851):
  zero = torch.tensor(0.0, dtype=r.dtype, device=r.device)
  unit = torch.tensor(1.0, dtype=r.dtype, device=r.device)
  c = t*t/6
  
  r_abs = torch.abs(r)
  tmp = 1 - torch.square(r_abs/t)
  tmp2 = tmp*tmp
  tmp3 = tmp*tmp2
  w = torch.where(r_abs < t, tmp2, zero)
  # rho = c * torch.where(r_abs < t, tmp3, unit)
  return w # rho