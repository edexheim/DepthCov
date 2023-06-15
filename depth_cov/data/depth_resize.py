import torch
import torch.nn.functional as nnf
import torchvision.transforms.functional as TF

# Note: Right now only handling factors of 2
def pyr_depth(depth, mode, kernel_size):
  stride = kernel_size

  if mode == "bilinear":
    new_depth = nnf.avg_pool2d(depth, kernel_size, stride)
  elif mode == "nearest_neighbor":
    new_depth = depth[:,:,0::stride,0::stride]
  elif mode == "max":
    new_depth = nnf.max_pool2d(depth, kernel_size)
  elif mode == "min":
    new_depth = -nnf.max_pool2d(-depth, kernel_size)
  elif mode == "masked_bilinear":
    mask = ~depth.isnan()
    depth_masked = torch.zeros_like(depth, device=depth.device)
    depth_masked[mask] = depth[mask]
    depth_sum = nnf.avg_pool2d(depth_masked, kernel_size, stride, divisor_override=1)
    mask_sum = nnf.avg_pool2d(mask.float(), kernel_size, stride, divisor_override=1)
    new_depth = torch.where(mask_sum > 0.0, depth_sum/mask_sum, torch.tensor(0.0, dtype=depth.dtype, device=depth.device))
  else:
    raise ValueError("pyr_depth mode: " + mode + " is not implemented.")

  return new_depth

def resize_depth(depth, mode, size):
  if mode == "bilinear":
    new_depth = TF.resize(depth, size, interpolation = TF.InterpolationMode.BILINEAR)  
  elif mode == "nearest_neighbor":
    new_depth = TF.resize(depth, size, interpolation = TF.InterpolationMode.NEAREST) 
  else:
    raise ValueError("resize_depth mode: " + mode + " is not implemented.")

  return new_depth

  # Note: Right now only handling factors of 2
def resize_depth_factor2(depth, mode, num_levels):
  kernel_size = 2**num_levels
  new_size = depth.size // (2**kernel_size)
  if mode == "bilinear":
    new_depth = TF.resize(depth, new_size, interpolation = TF.InterpolationMode.BILINEAR)  
  elif mode == "nearest_neighbor":
    new_depth = TF.resize(depth, new_size, interpolation = TF.InterpolationMode.NEAREST) 
  elif mode == "max":
    new_depth = nnf.max_pool2d(depth, kernel_size)
  elif mode == "min":
    new_depth = -nnf.max_pool2d(-depth, kernel_size)
  else:
    raise ValueError("resize_depth mode: " + mode + " is not implemented.")

  return new_depth