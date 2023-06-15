import torch
import torch.nn as nn

class BaseConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
    super(BaseConv, self).__init__()
    self.act = nn.LeakyReLU(inplace=True)

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding, 
        stride)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
        padding, stride)

  def forward(self, x):
    x = self.act(self.conv1(x))
    x = self.act(self.conv2(x))
    return x

class ResidualConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
    super(ResidualConv, self).__init__()
    self.act = nn.LeakyReLU(inplace=True)

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding, 
        stride)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
        padding, stride)

    self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)

    self.norm = nn.GroupNorm(16, out_channels)

  def forward(self, x):
    y = self.act(self.norm(self.conv1(x)))
    y = self.norm(self.conv2(y))
    x = self.conv3(x)
    return self.act(x + y)

class DownConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
    super(DownConv, self).__init__()

    self.pool1 = nn.MaxPool2d(kernel_size=2)
    self.conv_block = ResidualConv(in_channels, out_channels, kernel_size,
        padding, stride)

  def forward(self, x):
    x = self.pool1(x)
    x = self.conv_block(x)
    return x

class UpConv(nn.Module):
  def __init__(self, in_channels, in_channels_skip, out_channels,
      kernel_size, padding, stride):
    super(UpConv, self).__init__()

    # TODO: ConvTranspose2d vs. Upsample
    upsample_channels = in_channels//2

    assert upsample_channels == in_channels_skip

    if False:
      self.upsample = nn.ConvTranspose2d(
          in_channels, upsample_channels, kernel_size=2, padding=0, stride=2)
    else:
      self.upsample = nn.Sequential( \
        nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners = False), \
        nn.Conv2d(in_channels, upsample_channels, kernel_size=kernel_size, padding=padding, stride=stride)
      )

    self.conv_block = ResidualConv(
        in_channels=upsample_channels + in_channels_skip,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride)

  def forward(self, x, x_skip):
    x = self.upsample(x)
    x = torch.cat((x, x_skip), dim=1)
    x = self.conv_block(x)
    return x