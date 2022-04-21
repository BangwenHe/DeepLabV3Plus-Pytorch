import torch
import torch.nn as nn

from ._idaup import AtrousMobileNetUp


class CenterNetBranch(nn.Module):
    def __init__(self, heads, channels, out_dim=24, head_conv=False):
        super(CenterNetBranch, self).__init__()
        self.idaup = AtrousMobileNetUp(channels, out_dim)
        self.heads = heads

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(out_dim, head_conv,
                        kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, num_output, 
                        kernel_size=1, stride=1, padding=0))
            else:
                fc = nn.Conv2d(
                    in_channels=out_dim,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            setattr(self, head, fc)
    
    def forward(self, feats):
        x = self.idaup(feats)
        outs = {}
        for key in self.heads:
            outs[key] = getattr(self, key)(x)
        return outs
