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


@NECKS.register_module()
class NuScenesAtlasNeckV2(nn.Module):
    def __init__(self, in_channels, out_channels, n_down_layers, n_up_layers):
        super().__init__()
        self.down_layers = nn.ModuleList()
        self.laterals = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.out_convs = nn.ModuleList()
        self.n = len(n_down_layers)
        assert len(n_up_layers) == self.n - 1
        for i in range(self.n):
            n_channels = in_channels * (2 ** i)
            layer = []
            if i != 0:
                layer.append(get_conv_3d_3_2(n_channels // 2, n_channels))
            for _ in range(n_down_layers[i]):
                layer.append(get_basic_block_3d(n_channels, n_channels))
            self.down_layers.append(nn.Sequential(*layer))
            self.laterals.append(get_conv_2d_1(n_channels, out_channels))
        for i in range(self.n - 1):
            if i == self.n - 2:
                n_channels = in_channels * (2 ** (self.n - 1))
            else:
                n_channels = out_channels
            self.up_convs.append(get_conv_2d_1(n_channels, out_channels))
            layer = []
            for _ in range(n_up_layers[self.n - 2 - i]):
                layer.append(get_basic_block_2d(out_channels, out_channels))
            self.up_layers.append(nn.Sequential(*layer))
            self.out_convs.append(get_conv_2d_1(out_channels, out_channels))

    def forward(self, x):
        xs = []
        for layer in self.down_layers:
            x = layer(x)
            xs.append(x)
        assert x.shape[-1] == 1
        x = x[..., 0]
        outs = []
        for i in range(self.n - 1)[::-1]:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = self.up_convs[i](x)
            y = self.laterals[i](torch.mean(xs[i], dim=-1))
            x = (x + y) / 2
            x = self.up_layers[i](x)
            outs.append(self.out_convs[i](x).transpose(-1, -2))
        return outs[::-1]

    def init_weights(self):
        pass
