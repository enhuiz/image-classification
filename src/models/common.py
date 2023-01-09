import math

import torch
from einops import repeat
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
            x: (...)
        Returns:
            pe: (... d)
        """
        omega = self.omega
        while omega.dim() <= x.dim():
            omega = omega.unsqueeze(0)  # (... d)

        x = x.unsqueeze(-1)  # (... 1)
        x = omega * x
        x = torch.cat([x.sin(), x.cos()], dim=-1)

        return x

    def get_pe(self, n: int):
        """
        Args:
            n: int
        Returns:
            pe: (n d)
        """
        device = self.omega.device
        return self.forward(torch.arange(n, device=device))

    def add_pe_2d(self, x):
        """
        Args:
            x: (b c h w)
        """
        h = self.get_pe(x.shape[2])  # h d
        w = self.get_pe(x.shape[3])  # w d

        h = repeat(h, "h d -> b d h 1", b=len(x))
        w = repeat(w, "w d -> b d 1 w", b=len(x))

        x = x + h + w

        return x
