import torch
import torchvision.transforms.functional as TF

def init_gpu(gpu):
  for i in range(100):
    init_tensor = torch.zeros((8,192,256), device="cpu").to(gpu)
    del init_tensor

def str_to_dtype(str):
  if str == "float":
    return torch.float
  elif str == "double":
    return torch.double
  else:
    raise ValueError("Cannot convert : " + str + " to tensor type.")

def safe_sqrt(x):
  return torch.sqrt(x + 1e-8)

def swap_coords_xy(coords):
  coords_swap = torch.empty_like(coords)
  coords_swap[...,0] = coords[...,1]
  coords_swap[...,1] = coords[...,0]
  return coords_swap

# Transforms pixel coordiantes to [-1,1] assuming pixels are at fractional coordinates
def normalize_coordinates(x_pixel, dims):
  A = 1.0/torch.as_tensor(dims, device=x_pixel.device, dtype=x_pixel.dtype)
  x_norm = 2*A*x_pixel + A - 1
  return x_norm

def normalize_coordinates_A(x_pixel, A):
  x_norm = 2*A*x_pixel + A - 1  
  return x_norm

def unnormalize_coordinates(x_norm, dims):
  A = torch.as_tensor(dims, device=x_norm.device, dtype=x_norm.dtype)/2.0
  x_pixel = A*x_norm + A - 0.5
  return x_pixel

# Propagation of uncertainty using linear transform from pixel to normalized coordinates
def to_normalized_covariance(E_pixel, dims):
  A = torch.diag(2.0/torch.tensor(dims, device=E_pixel.device))
  E_norm = A @ E_pixel @ A.T
  return E_norm

# Propagation of uncertainty using linear transform from normalized to pixel coordinates
def to_pixel_covariance(E_norm, dims):
  B = torch.diag(torch.as_tensor(dims, device=E_norm.device, dtype=E_norm.dtype)/2.0)
  E_pixel = B @ E_norm @ B.T
  return E_pixel


def sample_coordinates(coords, dim, n):
  perm = torch.randperm(coords.shape[dim], device=coords.device)
  sample_ind = perm[:n]
  sampled_coords = torch.index_select(coords, dim, sample_ind)
  return sampled_coords

def downsample_depth(depth, scale_factor):
  pyr_size = torch.Size(torch.div(torch.tensor(depth.shape[-2:]), scale_factor, rounding_mode='floor'))
  depth_r = TF.resize(depth, size=pyr_size, interpolation=TF.InterpolationMode.NEAREST)
  return depth_r

def get_test_coords(img_size, device, batch_size=1):
  h,w = img_size
  y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
  test_coords = torch.column_stack((torch.flatten(y_coords), torch.flatten(x_coords)))
  test_coords = test_coords.repeat(batch_size,1,1)
  return test_coords

def get_coord_img(img_size, device, batch_size=1):
  h,w = img_size
  y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
  coord_img = torch.dstack((y_coords, x_coords)).unsqueeze(0).repeat(batch_size,1,1,1)
  return coord_img

def fill_image(coords, vals, img_size, default_val = float('nan')):
  coords_long = coords.long()
  img = default_val*torch.ones((1,img_size[0],img_size[1]), device=coords.device, dtype=vals.dtype)
  img[:,coords_long[...,0],coords_long[...,1]] = vals[...,0]
  return img

def bilinear_interpolation(img, x):

  # x coordinates are normalized [-1,1]
  # NOTE: grid_sample expects (x,y) as in image coordinates (so column then row)
  x_samples = torch.unsqueeze(x, dim=1)
  ind_swap = torch.tensor([1, 0], device=img.device)
  x_samples = torch.index_select(x_samples, 3, ind_swap)

  # kernel_image shape: B x C x H x W
  # x shape: B x N x 2
  # output shape: B x C x N
  B = img.shape[0]
  N = x.shape[1]
  rows = img.shape[2]
  cols = img.shape[3]

  # Get sampled features
  sampled_params = torch.nn.functional.grid_sample(img, x_samples, 
    mode='bilinear', padding_mode='reflection', align_corners=False)
  sampled_params = torch.permute(torch.squeeze(sampled_params, dim=2), (0,2,1))

  return sampled_params


def sample_coords(depth, gaussian_cov_params, num_samples, mode):
  B = depth.shape[0]
  H = depth.shape[-2]
  W = depth.shape[-1]
  device = depth.device
  y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
  coord_img = torch.dstack((y_coords, x_coords)).unsqueeze(0).repeat(B,1,1,1)

  depth_vec = torch.reshape(depth, (B,-1))
  valid_depths = ~depth_vec.isnan() 
  # Take minimum so that batch sizes are equal
  num_valid = torch.min(torch.count_nonzero(valid_depths, dim=1))
  coord_vec = torch.reshape(coord_img, (B,-1,2))

  num_coord_samples = num_valid
  if num_coord_samples > num_samples:
    num_coord_samples = num_samples
  
  if mode == "uniform":
    weights = 1.0*valid_depths
  elif mode == "scale":
    scale_vec = torch.reshape(gaussian_cov_params[:,0,:,:], (B,-1))
    weights = ((1.0/scale_vec) + 1e-6) * valid_depths
  else:
    raise ValueError("sample_coords mode: " + mode + " is not implemented.")

  if num_coord_samples > 0:
    inds = torch.multinomial(weights, num_coord_samples, replacement=False)
    batch_inds = torch.arange(B,device=device).unsqueeze(1).repeat(1,num_coord_samples)
    coord_samples = coord_vec[batch_inds, inds, :] 
    depth_samples = depth_vec[batch_inds, inds]
    depth_samples = depth_samples.unsqueeze(-1)
  else:
    return None, None, None

  return coord_samples, depth_samples, batch_inds