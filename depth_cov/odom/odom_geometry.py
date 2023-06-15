import torch
from depth_cov.utils.lie_algebra import invertSE3, skew_symmetric

def get_T_w_curr(T_w_ref, T_curr_ref):
  T_w_curr = torch.matmul(T_w_ref, invertSE3(T_curr_ref))
  return T_w_curr

def get_rel_pose(pose1, pose2):
  T_12 = torch.matmul(invertSE3(pose1), pose2)
  return T_12

# w means global reference for affine parameters
def get_aff_w_curr(aff_w_ref, aff_curr_ref):
  aff_params_new = aff_w_ref.clone()
  aff_params_new[:,0,:] += aff_curr_ref[:,0,:]
  aff_params_new[:,1,:] += aff_curr_ref[:,1,:]*torch.exp(aff_curr_ref[:,0,:])
  return aff_params_new

def get_rel_aff(aff1, aff2):
  aff_rel = torch.empty_like(aff1)
  aff_rel[:,0,:] = aff1[:,0,:] - aff2[:,0,:]
  aff_rel[:,1,:] = torch.exp(-aff_rel[:,0,:]) * (aff1[:,1,:] - aff2[:,1,:])
  return aff_rel

# TODO: Is this correct? Could have subpixel errors
def resize_intrinsics(K, image_scale_factors):
  T = torch.tensor([[image_scale_factors[1], 0, 0.5*image_scale_factors[1]-0.5],
                    [0, image_scale_factors[0], 0.5*image_scale_factors[0]-0.5],
                    [0, 0, 1]], device=K.device, dtype=K.dtype)
  # T = torch.tensor([[image_scale_factors[1], 0, image_scale_factors[1]],
  #                   [0, image_scale_factors[0], image_scale_factors[0]],
  #                   [0, 0, 1]])
  K_new = torch.matmul(T,K)
  return K_new

def get_mean_log_depth(depth):
  mask = ~depth.isnan()
  valid_depths = depth[mask].unsqueeze(0).unsqueeze(-1)
  valid_log_depths = torch.log(valid_depths)
  mean_log_depth = torch.mean(valid_log_depths, dim=(1,2), keepdim=True)
  return mean_log_depth

def get_log_median_depth(depth):
  mask = ~depth.isnan()
  valid_depths = depth[mask].unsqueeze(0).unsqueeze(-1)
  median_depth = torch.median(valid_depths, dim=1, keepdim=True)
  log_median_depths = torch.log(median_depth.values)
  return log_median_depths

# log_depth_train (B, N_train, 1)
# mean_log_depth (B, 1, 1)
# Knm_Kmminv (B, N_test, N_train)
def predict_log_depth(log_depth_train, mean_log_depth, Knm_Kmminv):

  log_depth_test = mean_log_depth + torch.matmul(Knm_Kmminv, log_depth_train - mean_log_depth)
  dlogz_dd = Knm_Kmminv
  dlogz_ds = 1.0 - torch.sum(Knm_Kmminv, dim=2, keepdim=True)
  return log_depth_test, dlogz_dd.unsqueeze(-2), dlogz_ds.unsqueeze(-2)

def predict_log_depth_img_no_J(log_depth_train, mean_log_depth, Knm_Kmminv):
  b, h, w, m = Knm_Kmminv.shape
  log_depth_test = (mean_log_depth + torch.matmul(
      Knm_Kmminv.view(b,h*w,m), log_depth_train - mean_log_depth)).view(b,h,w,1)
  return log_depth_test

def predict_log_depth_img(log_depth_train, mean_log_depth, Knm_Kmminv):
  b, h, w, m = Knm_Kmminv.shape
  log_depth_test = (mean_log_depth + torch.matmul(
      Knm_Kmminv.view(b,h*w,m), log_depth_train - mean_log_depth)).view(b,h,w,1)
  dlogz_dd = Knm_Kmminv
  dlogz_ds = 1.0 - torch.sum(Knm_Kmminv, dim=-1, keepdim=True)
  return log_depth_test, dlogz_dd.unsqueeze(-2), dlogz_ds.unsqueeze(-2)

# log_depths (B, N, 1)
def log_depth_to_depth(log_depth):
  depth = torch.exp(log_depth)
  dd_dlogd = depth.unsqueeze(-1)
  return depth, dd_dlogd

def depth_to_log_depth(depth):
  log_depth = torch.log(depth)
  dlogd_dd = (1.0/depth).unsqueeze(-1)
  return log_depth, dlogd_dd


# K is (3, 3)
# P is (B, N, 3)
def projection(K, P):
  tmp1 = K[0,0] * P[...,0]/P[...,2]
  tmp2 = K[1,1] * P[...,1]/P[...,2]

  p = torch.empty(P.shape[:-1] + (2,), device=P.device, dtype=K.dtype)
  p[...,0] = tmp1 + K[0,2]
  p[...,1] = tmp2 + K[1,2]

  dp_dP = torch.empty(P.shape[:-1] + (2,3), device=P.device, dtype=K.dtype)
  dp_dP[...,0,0] = K[0,0]
  dp_dP[...,0,1] = 0.0
  dp_dP[...,0,2] = -tmp1
  dp_dP[...,1,0] = 0.0
  dp_dP[...,1,1] = K[1,1]
  dp_dP[...,1,2] = -tmp2
  dp_dP /= P[...,2,None,None]

  return p, dp_dP

# K is (3, 3)
# P is (..., 3)
def projectionNoJ(K, P):
  p = torch.empty(P.shape[:-1] + (2,), device=P.device, dtype=K.dtype)
  p[...,0] = K[0,0] * P[...,0]/P[...,2] + K[0,2]
  p[...,1] = K[1,1] * P[...,1]/P[...,2] + K[1,2]

  return p

# T is (B, 4, 4)
# P is (B, N, 3)
def transformPointsNoJ(Tji, Pi):
  R = Tji[:,None,:3,:3].contiguous()
  t = Tji[:,None,:3,3:4].contiguous()

  Pj = torch.matmul(R, Pi[...,None]) + t
  Pj = torch.squeeze(Pj, dim=-1)
  return Pj


def transform_project(K, Tji, Pi):
  Pmat = torch.matmul(K[None,:,:], Tji[:,0:3,:])

  A = Pmat[:,None,:3,:3].contiguous()
  b = Pmat[:,None,:3,3:4].contiguous()

  p_h = torch.squeeze(torch.matmul(A, Pi[...,None]) + b, dim=-1)

  depth = p_h[:,:,2:3]
  coords = p_h[:,:,:2]/depth

  return coords, depth
  

# K is (3, 3)
# p is (B, N, 2)
# z is (B, N, 1)
def backprojection(K, p, z):
  tmp1 = (p[...,0] - K[0,2])/K[0,0]
  tmp2 = (p[...,1] - K[1,2])/K[1,1]

  dP_dz = torch.empty(p.shape[:-1]+(3,1), device=z.device, dtype=K.dtype)
  dP_dz[...,0,0] = tmp1
  dP_dz[...,1,0] = tmp2
  dP_dz[...,2,0] = 1.0

  P = torch.squeeze(z[...,None,:]*dP_dz, dim=-1)
  
  return P, dP_dz

# T is (B, 4, 4)
def adjoint_matrix(T):
  B = T.shape[0]
  adj_mat = torch.empty((B,6,6), device=T.device, dtype=T.dtype)
  adj_mat[:,:3,:3] = T[:,:3,:3]
  adj_mat[:,:3,3:] = 0.0
  adj_mat[:,3:,:3] = torch.matmul(skew_symmetric(T[:,:3,3]), T[:,:3,:3])
  adj_mat[:,3:,3:] = T[:,:3,:3]
  return adj_mat

def transformPoints(Tji, Pi):
  R = Tji[:,None,:3,:3].contiguous()
  t = Tji[:,None,:3,3:4].contiguous()

  Pj = torch.matmul(R, Pi[...,None]) + t
  Pj = torch.squeeze(Pj, dim=-1)

  dPj_dT = torch.empty(Pi.shape[:-1] + (3, 6), device=Pi.device, dtype=Tji.dtype)
  dPj_dT[...,:,:3] = -torch.matmul(R, skew_symmetric(Pi))
  dPj_dT[...,:,3:] = R

  dPj_dPi = R

  return Pj, dPj_dT, dPj_dPi

def transformPointsImg(Tji, Pi):
  R = Tji[:,None,None,:3,:3].contiguous()
  t = Tji[:,None,None,:3,3:4].contiguous()

  Pj = torch.matmul(R, Pi[...,None]) + t
  Pj = torch.squeeze(Pj, dim=-1)


  dPj_dT = torch.empty(Pi.shape[:-1] + (3, 6), device=Pi.device, dtype=Tji.dtype)
  dPj_dT[...,:,:3] = -torch.matmul(R, skew_symmetric(Pi))
  dPj_dT[...,:,3:] = R

  dPj_dPi = R

  return Pj, dPj_dT, dPj_dPi


# T0 is (B, 4, 4)
# T1 is (B, 4, 4)
# T1_inv @ T0
def between(T0, T1):
  B = T0.shape[0]
  T1_inv = invertSE3(T1)
  T10 = torch.matmul(T1_inv, T0)

  dT10_dT0 = torch.eye(6,device=T0.device,dtype=T0.dtype).unsqueeze(0).repeat((B,1,1))
  T0_inv = invertSE3(T0)
  dT10_dT1 = -torch.matmul(adjoint_matrix(T0_inv), adjoint_matrix(T1))

  return T10, dT10_dT0, dT10_dT1