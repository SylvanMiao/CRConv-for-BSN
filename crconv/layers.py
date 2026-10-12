import math

import torch
from torch import nn


class CRConv2d(nn.Conv2d):
    """Shared blind/visible convolution with a parameter-tied center."""

    def __init__(self, *args, center_mode="kernel_softmax", temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if any(k < 3 or k % 2 == 0 for k in self.kernel_size):
            raise ValueError("CRConv requires odd kernel dimensions >= 3")
        if self.padding_mode != "zeros":
            raise ValueError("Only zero padding is supported by this BSN example")
        if center_mode not in {"kernel_softmax", "tied_mean", "harmonic_4"}:
            raise ValueError("Unknown center completion rule")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.center_mode = center_mode
        self.temperature = float(temperature)
        mask = torch.ones_like(self.weight)
        mask[..., self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
        self.register_buffer("mask", mask)

    def effective_weight(self, blind=True):
        w = self.weight * self.mask
        if blind:
            return w
        count = self.mask.sum((-2, -1), keepdim=True)
        if self.center_mode == "tied_mean":
            center = w.sum((-2, -1), keepdim=True) / count
        elif self.center_mode == "harmonic_4":
            y, x = self.kernel_size[0] // 2, self.kernel_size[1] // 2
            center = (w[..., y - 1, x] + w[..., y + 1, x]
                      + w[..., y, x - 1] + w[..., y, x + 1])[..., None, None] / 4
        else:
            scale = w.abs().sum((-2, -1), keepdim=True) / count
            logits = w.abs() / scale.clamp_min(1e-12) / self.temperature
            logits = logits.masked_fill(self.mask == 0, -torch.inf)
            weights = logits.flatten(-2).softmax(-1).view_as(w)
            center = (weights * w).sum((-2, -1), keepdim=True)
        return w + (1 - self.mask) * center

    def forward(self, x, blind=True):
        return self._conv_forward(x, self.effective_weight(blind), self.bias)
