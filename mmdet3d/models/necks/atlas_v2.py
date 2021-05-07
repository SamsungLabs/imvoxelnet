import torch
from torch import nn
import torch.nn.functional as F
from mmdet.models import NECKS


class BasicBlock(nn.Module):
    def __init__(self, conv, bn, in_channels, out_channels):
        super().__init__()
        self.conv1 = conv(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = bn(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = bn(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


def get_basic_block_3d(*args, **kwargs):
    return BasicBlock(nn.Conv3d, nn.BatchNorm3d, *args, **kwargs)


def get_basic_block_2d(*args, **kwargs):
    return BasicBlock(nn.Conv2d, nn.BatchNorm2d, *args, **kwargs)


def get_conv(conv, bn, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        conv(in_channels, out_channels, kernel_size, stride, padding),
        bn(out_channels),
        nn.ReLU(inplace=True)
    )


def get_conv_3d(*args, **kwargs):
    return get_conv(nn.Conv3d, nn.BatchNorm3d, *args, **kwargs)


def get_conv_2d(*args, **kwargs):
    return get_conv(nn.Conv2d, nn.BatchNorm2d, *args, **kwargs)


def get_conv_3d_1(in_channels, out_channels):
    return get_conv_3d(in_channels, out_channels, 1, 1, 0)


def get_conv_3d_3_2(in_channels, out_channels):
    return get_conv_3d(in_channels, out_channels, 3, 2, 1)


def get_conv_2d_1(in_channels, out_channels):
    return get_conv_2d(in_channels, out_channels, 1, 1, 0)
