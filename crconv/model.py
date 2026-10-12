import torch
from torch import nn

from .layers import CRConv2d


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return x + self.body(x)


class BlindBranch(nn.Module):
    def __init__(self, channels, dilation, blocks, center_mode, temperature):
        super().__init__()
        self.center_conv = CRConv2d(
            channels, channels, 2 * dilation - 1, padding=dilation - 1,
            center_mode=center_mode, temperature=temperature)
        self.body = nn.Sequential(
            nn.GELU(), nn.Conv2d(channels, channels, 1), nn.GELU(),
            nn.Conv2d(channels, channels, 1), nn.GELU(),
            *[DilatedResidualBlock(channels, dilation) for _ in range(blocks)],
            nn.Conv2d(channels, channels, 1), nn.GELU())

    def forward(self, x, blind=True):
        return self.body(self.center_conv(x, blind=blind))


class DBSN(nn.Module):
    """No-PD two-branch backbone used in the project's D-BSN experiments."""

    def __init__(self, in_channels=1, out_channels=1, base_channels=128, blocks=9,
                 center_mode="kernel_softmax", temperature=1.0):
        super().__init__()
        if base_channels < 2 or base_channels % 2 or blocks < 0:
            raise ValueError("Use positive even base_channels and nonnegative blocks")
        self.head = nn.Sequential(nn.Conv2d(in_channels, base_channels, 1), nn.GELU())
        self.branch_d2 = BlindBranch(base_channels, 2, blocks, center_mode, temperature)
        self.branch_d3 = BlindBranch(base_channels, 3, blocks, center_mode, temperature)
        self.output = nn.Sequential(
            nn.Conv2d(2 * base_channels, 2 * base_channels, 1), nn.GELU(),
            nn.Conv2d(2 * base_channels, 2 * base_channels, 1), nn.GELU(),
            nn.Conv2d(2 * base_channels, out_channels, 1))

    def forward(self, x, blind=True):
        x = self.head(x)
        return self.output(torch.cat((self.branch_d2(x, blind), self.branch_d3(x, blind)), 1))
