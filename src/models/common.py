import math

import torch
from torch import nn


class SinusodialEmbedding(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        exponent = torch.arange(self.half_dim, dtype=torch.float32)
        exponent = exponent / self.half_dim
        omega = torch.exp(-math.log(1e4) * exponent)
        self.omega: torch.Tensor
        self.register_buffer("omega", omega, persistent=False)

    @property
    def half_dim(self):
        assert self.num_channels % 2 == 0, "Only support even num_channels."
        return self.num_channels // 2

    def forward(self, x):
        """
        Args:
            t: (b t)
        Returns:
            pe: (b d t)
        """
        assert x.dim() == 2
        e = self.omega[None, :, None] * x.unsqueeze(1)
        e = torch.cat([e.sin(), e.cos()], dim=1)
        return e

    def add_pe_2d(self, x):
        """
        Args:
            x: (b c h w)
        """
        h = self.forward(torch.arange(x.shape[2])[None].to(x.device))
        w = self.forward(torch.arange(x.shape[3])[None].to(x.device))

        h = h.unsqueeze(3)  # b c h 1
        w = w.unsqueeze(2)  # b c 1 h

        x = x + h + w

        return x
