import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
import depth_cov.data.depth_resize as depth_resize


def add_gaussian_noise(tensor, mean, std):
  return tensor + torch.randn(tensor.size()) * std + mean

class TrainTransform():
  def __init__(self, size, max_angle, crop_scale):
    self.size = size
    self.max_angle = max_angle

    if crop_scale:
      self.random_resized_crop = T.RandomResizedCrop(self.size, scale=(0.5,1.0), ratio=(1.0,1.0), interpolation=TF.InterpolationMode.BILINEAR)
    else:
      self.random_resized_crop = T.RandomResizedCrop(self.size, scale=(1.0,1.0), ratio=(1.0,1.0), interpolation=TF.InterpolationMode.BILINEAR)

    self.color_jitter_tf = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

  def __call__(self, img, depth):

    i, j, h, w = self.random_resized_crop.get_params(img, self.random_resized_crop.scale, self.random_resized_crop.ratio)
    img_cropped = TF.resized_crop(img, i, j, h, w, self.random_resized_crop.size, TF.InterpolationMode.BILINEAR)
    depth_cropped = TF.resized_crop(depth, i, j, h, w, self.random_resized_crop.size, TF.InterpolationMode.NEAREST)

    should_flip = (torch.rand(1).item() < 0.5)
    if should_flip:
      img_cropped = TF.hflip(img_cropped)
      depth_cropped = TF.hflip(depth_cropped)

    img_adj = self.color_jitter_tf(img_cropped)

    angle = random.randint(-self.max_angle, self.max_angle)
    img_rot = TF.rotate(img_adj, angle, interpolation = TF.InterpolationMode.BILINEAR, fill=-1e3)
    depth_rot = TF.rotate(depth_cropped, angle, interpolation = TF.InterpolationMode.NEAREST, fill=-1e3)  

    img_rot[img_rot<0] = 0.0
    depth_rot[depth_rot<0] = float('nan')

    return img_rot, depth_rot
    
class BaseTransform():
  def __init__(self, size):
    self.size = size
  
  def __call__(self, img, depth):
    img_r = TF.resize(img, self.size, interpolation = TF.InterpolationMode.BILINEAR, antialias = True)
    depth_r = depth_resize.resize_depth(depth, mode="nearest_neighbor", size=self.size)
    return img_r, depth_r