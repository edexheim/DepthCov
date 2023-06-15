import torch

def det2x2(mats):
  dets = mats[...,0,0] * mats[...,1,1] - mats[...,0,1] * mats[...,1,0]
  return dets

def trace2x2(mats):
  return mats[...,0,0] + mats[...,1,1]

def inv2x2(mats):
  invs = torch.empty_like(mats)
  invs[...,0,0] = mats[...,1,1]
  invs[...,1,1] = mats[...,0,0]
  invs[...,0,1] = -mats[...,1,0]
  invs[...,1,0] = -mats[...,0,1]

  determinants = det2x2(mats)

  invs *= (1.0/determinants[...,None,None])
  
  return invs, determinants

def cholesky2x2(mats, upper=True):
  chol = torch.empty_like(mats)
  if upper:
    chol[...,1,0] = 0
    chol[...,0,0] = torch.sqrt(mats[...,0,0])
    chol[...,0,1] = torch.div(mats[...,1,0], chol[...,0,0])
    chol[...,1,1] = torch.sqrt(mats[...,1,1] - torch.square(chol[...,0,1]))
  else:
    chol[...,0,1] = 0
    chol[...,0,0] = torch.sqrt(mats[...,0,0])
    chol[...,1,0] = torch.div(mats[...,1,0], chol[...,0,0])
    chol[...,1,1] = torch.sqrt(mats[...,1,1] - torch.square(chol[...,1,0]))
  
  return chol

def chol_log_det(L):
  return 2*torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=1)

def trace(A):
  return torch.sum(torch.diagonal(A, dim1=-2, dim2=-1), dim=1)