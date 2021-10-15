import torch
from torch import nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models import NECKS


@NECKS.register_module()
class FastIndoorImVoxelNeck(nn.Module):
    def __init__(self, in_channels, n_blocks, out_channels):
        super(FastIndoorImVoxelNeck, self).__init__()
        self.n_scales = len(n_blocks)
        n_channels = in_channels
        for i in range(len(n_blocks)):
            stride = 1 if i == 0 else 2
            self.__setattr__(f'down_layer_{i}', self._make_layer(stride, n_channels, n_blocks[i]))
            n_channels = n_channels * stride
            if i > 0:
                self.__setattr__(f'up_block_{i}', self._make_up_block(n_channels, n_channels // 2))
            self.__setattr__(f'out_block_{i}', self._make_block(n_channels, out_channels))

    def forward(self, x):
        down_outs = []
        for i in range(self.n_scales):
            x = self.__getattr__(f'down_layer_{i}')(x)
            down_outs.append(x)
        outs = []
        for i in range(self.n_scales - 1, -1, -1):
            if i < self.n_scales - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = down_outs[i] + x
            out = self.__getattr__(f'out_block_{i}')(x)
            outs.append(out)
        return outs[::-1]

    @staticmethod
    def _make_layer(stride, n_channels, n_blocks):
        blocks = []
        for i in range(n_blocks):
            if i == 0 and stride != 1:
                blocks.append(BasicBlock3dV2(n_channels, n_channels * 2, stride))
                n_channels = n_channels * 2
            else:
                blocks.append(BasicBlock3dV2(n_channels, n_channels))
        return nn.Sequential(*blocks)

    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _make_up_block(in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, 2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def init_weights(self):
        pass


@NECKS.register_module()
class ImVoxelNeck(nn.Module):
    def __init__(self, channels, out_channels, down_layers, up_layers, conditional):
        super().__init__()
        self.model = EncoderDecoder(channels=channels,
                                    layers_down=down_layers,
                                    layers_up=up_layers,
                                    cond_proj=conditional)
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_channels in channels])

    @auto_fp16()
    def forward(self, x):
        x = self.model.forward(x)[::-1]
        return [self.conv_blocks[i](x[i]) for i in range(len(x))]

    def init_weights(self):
        pass


@NECKS.register_module()
class KittiImVoxelNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            BasicBlock3d(in_channels, in_channels),
            self._get_conv(in_channels, in_channels * 2),
            BasicBlock3d(in_channels * 2, in_channels * 2),
            self._get_conv(in_channels * 2, in_channels * 4),
            BasicBlock3d(in_channels * 4, in_channels * 4),
            # todo: padding should be (1, 1, 0) here
            self._get_conv(in_channels * 4, out_channels, 1, 0)
        )

    @staticmethod
    def _get_conv(in_channels, out_channels, stride=(1, 1, 2), padding=(1, 1, 1)):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    @auto_fp16()
    def forward(self, x):
        x = self.model.forward(x)
        assert x.shape[-1] == 1
        return [x[..., 0].transpose(-1, -2)]

    def init_weights(self):
        pass


@NECKS.register_module()
class NuScenesImVoxelNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            BasicBlock3d(in_channels, in_channels),
            self._get_conv(in_channels, in_channels * 2, 2, 1),
            BasicBlock3d(in_channels * 2, in_channels * 2),
            self._get_conv(in_channels * 2, in_channels * 4),
            BasicBlock3d(in_channels * 4, in_channels * 4),
            self._get_conv(in_channels * 4, out_channels, 1, (1, 1, 0))
        )

    @staticmethod
    def _get_conv(in_channels, out_channels, stride=(1, 1, 2), padding=(1, 1, 1)):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    @auto_fp16()
    def forward(self, x):
        x = self.model.forward(x)
        assert x.shape[-1] == 1
        return [x[..., 0].transpose(-1, -2)]

    def init_weights(self):
        pass


# Everything below is copied from https://github.com/magicleap/Atlas/blob/master/atlas/backbone3d.py
def get_norm_3d(norm, out_channels):
    """ Get a normalization module for 3D tensors
    Args:
        norm: (str or callable)
        out_channels
    Returns:
        nn.Module or None: the normalization layer
    """

    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm3d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm(out_channels)


def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class BasicBlock3d(nn.Module):
    """ 3x3x3 Resnet Basic Block"""
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm='BN', drop=0):
        super(BasicBlock3d, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3(inplanes, planes, stride, 1, dilation)
        self.bn1 = get_norm_3d(norm, planes)
        self.drop1 = nn.Dropout(drop, True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, 1, 1, dilation)
        self.bn2 = get_norm_3d(norm, planes)
        self.drop2 = nn.Dropout(drop, True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.drop1(out) # drop after both??
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop2(out) # drop after both??

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock3dV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3dV2, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm3d(out_channels)
        if self.stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.stride != 1:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ConditionalProjection(nn.Module):
    """ Applies a projected skip connection from the encoder to the decoder
    When condition is False this is a standard projected skip connection
    (conv-bn-relu).
    When condition is True we only skip the non-masked features
    from the encoder. To maintin scale we instead skip the decoder features.
    This was intended to reduce artifacts in unobserved regions,
    but was found to not be helpful.
    """

    def __init__(self, n, norm='BN', condition=True):
        super(ConditionalProjection, self).__init__()
        # return relu(bn(conv(x)) if mask, relu(bn(y)) otherwise
        self.conv = conv1x1x1(n, n)
        self.norm = get_norm_3d(norm, n)
        self.relu = nn.ReLU(True)
        self.condition = condition

    def forward(self, x, y, mask):
        """
        Args:
            x: tensor from encoder
            y: tensor from decoder
            mask
        """

        x = self.conv(x)
        if self.condition:
            x = torch.where(mask, x, y)
        x = self.norm(x)
        x = self.relu(x)
        return x


class EncoderDecoder(nn.Module):
    """ 3D network to refine feature volumes"""

    def __init__(self, channels=[32,64,128], layers_down=[1,2,3],
                 layers_up=[3,3,3], norm='BN', drop=0, zero_init_residual=True,
                 cond_proj=True):
        super(EncoderDecoder, self).__init__()

        self.cond_proj = cond_proj

        self.layers_down = nn.ModuleList()
        self.proj = nn.ModuleList()

        self.layers_down.append(nn.Sequential(*[
            BasicBlock3d(channels[0], channels[0], norm=norm, drop=drop)
            for _ in range(layers_down[0]) ]))
        self.proj.append( ConditionalProjection(channels[0], norm, cond_proj) )
        for i in range(1,len(channels)):
            layer = [nn.Conv3d(channels[i-1], channels[i], 3, 2, 1, bias=(norm=='')),
                     get_norm_3d(norm, channels[i]),
                     nn.Dropout(drop, True),
                     nn.ReLU(inplace=True)]
            layer += [BasicBlock3d(channels[i], channels[i], norm=norm, drop=drop)
                      for _ in range(layers_down[i])]
            self.layers_down.append(nn.Sequential(*layer))
            if i<len(channels)-1:
                self.proj.append( ConditionalProjection(channels[i], norm, cond_proj) )

        self.proj = self.proj[::-1]

        channels = channels[::-1]
        self.layers_up_conv = nn.ModuleList()
        self.layers_up_res = nn.ModuleList()
        for i in range(1,len(channels)):
            self.layers_up_conv.append( conv1x1x1(channels[i-1], channels[i]) )
            self.layers_up_res.append(nn.Sequential( *[
                BasicBlock3d(channels[i], channels[i], norm=norm, drop=drop)
                for _ in range(layers_up[i-1]) ]))

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each
        # residual block behaves like an identity. This improves the
        # model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock3d):
                    nn.init.constant_(m.bn2.weight, 0)


    def forward(self, x):
        if self.cond_proj:
            valid_mask = (x!=0).any(1, keepdim=True).float()


        xs = []
        for layer in self.layers_down:
            x = layer(x)
            xs.append(x)

        xs = xs[::-1]
        out = []
        for i in range(len(self.layers_up_conv)):
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
            x = self.layers_up_conv[i](x)
            if self.cond_proj:
                scale = 1/2**(len(self.layers_up_conv)-i-1)
                mask = F.interpolate(valid_mask, scale_factor=scale)!=0
            else:
                mask = None
            y = self.proj[i](xs[i+1], x, mask)
            x = (x + y)/2
            x = self.layers_up_res[i](x)

            out.append(x)

        return out
