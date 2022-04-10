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
