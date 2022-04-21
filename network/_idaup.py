"""
基于 CenterNet 的结构, 实现对Mobilenet输出的上采样. 

有两种实现方式: 1. 对feature map进行连续的上采样; 2. 合并原来相同层的feature map, 再与当前上采样后的feature map进行相加
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck


# copyright: https://github.com/CaoWGG/Mobilenetv2-CenterNet


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):
    def __init__(self, out_dim, channel):
        super(IDAUp, self).__init__()
        self.out_dim = out_dim
        self.up = nn.Sequential(
                    nn.ConvTranspose2d(
                        out_dim, out_dim, kernel_size=2, stride=2, padding=0,
                        output_padding=0, groups=out_dim, bias=False),
                    nn.BatchNorm2d(out_dim,eps=0.001,momentum=0.1),
                    nn.ReLU())
        self.conv =  nn.Sequential(
                    nn.Conv2d(channel, out_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dim,eps=0.001,momentum=0.1),
                    nn.ReLU(inplace=True))

    def forward(self, layers):
        layers = list(layers)
        x = self.up(layers[0])
        y = self.conv(layers[1])
        out = x + y
        return out


class MobileNetUp(nn.Module):
    def __init__(self, channels, out_dim = 24):
        super(MobileNetUp, self).__init__()
        channels =  channels[::-1]
        self.conv =  nn.Sequential(
                    nn.Conv2d(channels[0], out_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dim,eps=0.001,momentum=0.1),
                    nn.ReLU(inplace=True))
        self.conv_last =  nn.Sequential(
                    nn.Conv2d(out_dim,out_dim,
                              kernel_size=3, stride=1, padding=1 ,bias=False),
                    nn.BatchNorm2d(out_dim,eps=1e-5,momentum=0.01),
                    nn.ReLU(inplace=True))

        for i,channel in enumerate(channels[1:]):
            setattr(self,'up_%d'%(i),IDAUp(out_dim,channel))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.ConvTranspose2d):
                fill_up_weights(m)

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        x = self.conv(layers[-1])
        for i in range(0,len(layers)-1):
            up = getattr(self, 'up_{}'.format(i))
            x = up([x,layers[len(layers)-2-i]])
        x = self.conv_last(x)
        return x


# copyright: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/msra_resnet.py
BN_MOMENTUM = 0.1


def _get_deconv_cfg(deconv_kernel, index):
    if deconv_kernel == 4:
        padding = 1
        output_padding = 0
    elif deconv_kernel == 3:
        padding = 1
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 0
        output_padding = 0

    return deconv_kernel, padding, output_padding


def _make_deconv_layer(num_layers, num_filters, num_kernels, inplanes, deconv_with_bias=False):
    assert num_layers == len(num_filters), \
        'ERROR: num_deconv_layers is different len(num_deconv_filters)'
    assert num_layers == len(num_kernels), \
        'ERROR: num_deconv_layers is different len(num_deconv_filters)'

    layers = []
    for i in range(num_layers):
        kernel, padding, output_padding = \
            _get_deconv_cfg(num_kernels[i], i)

        planes = num_filters[i]
        layers.append(
            nn.ConvTranspose2d(
                in_channels=inplanes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=deconv_with_bias))
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))
        inplanes = planes

    return nn.Sequential(*layers)


# ==================================================================


class AtrousMobileNetUp(nn.Module):
    """
    因为DeeplabV3Plus使用了AtrousConvolution, 导致feature map分辨率不发生变化
    所以这里要使用两次BasicBlock来降低分辨率, 并融合两次的特征图
    """
    def __init__(self, channels, out_dim=24):
        # channels: [160, 320, 640, 1280]
        super(AtrousMobileNetUp, self).__init__()

        self.downsample_times = 2
        for i in range(self.downsample_times):
            setattr(self, 'downsample_%d'%(i), 
                BasicBlock(channels[i], 
                            channels[i+1], 
                            stride=2,
                            downsample=nn.Sequential(
                                nn.Conv2d(channels[i], channels[i+1],
                                          kernel_size=1, stride=2, bias=False),
                                nn.BatchNorm2d(channels[i+1], momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True))))
        
        channels =  channels[::-1]
        self.conv = nn.Sequential(
            nn.Conv2d(channels[0], out_dim,
                        kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
            nn.ReLU(inplace=True))

        self.upsample_times = self.downsample_times
        for i in range(self.upsample_times):
            setattr(self, 'upsample_%d'%(i), IDAUp(out_dim, channels[i+1]))
        
        self.alone_upsample = _make_deconv_layer(1, [out_dim], [3], out_dim)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.ConvTranspose2d):
                fill_up_weights(m)
    
    def forward(self, x):
        downsample_tensors = [x]
        for i in range(self.downsample_times):
            x = getattr(self, 'downsample_{}'.format(i))(x)
            downsample_tensors.append(x)

        x = self.conv(x)
        downsample_tensors = downsample_tensors[::-1]
        for i in range(self.upsample_times):
            x = getattr(self, 'upsample_{}'.format(i))([x, downsample_tensors[i+1]])
        x = self.alone_upsample(x)
        return x


if __name__ == "__main__":
    model = AtrousMobileNetUp([320,640,1280])
    print(model)
    x = torch.randn(1, 320, 64, 64)
    y = model(x)
    print(y.shape)
