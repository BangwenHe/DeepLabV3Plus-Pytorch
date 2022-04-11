import torch
from torch import nn
from torch.nn import functional as F


class MeanDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(MeanDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )
        self._init_weight_()

    def forward(self, x):
        return self.discriminator(x.mean(dim=(2, 3)))

    def _init_weight_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class ConvDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(ConvDiscriminator, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.discriminator = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self._init_weight_()

    def forward(self, x):
        x = self.convs(x)
        x = torch.mean(x, dim=(2, 3))
        return self.discriminator(x)

    def _init_weight_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
