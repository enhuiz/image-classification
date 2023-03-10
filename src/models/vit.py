import torch
from einops import rearrange, repeat
from torch import nn

from .common import SinusodialEmbedding


class ViT(nn.Module):
    def __init__(
        self,
        input_channels=3,
        patch_size=8,  # not using 16x16 as our image is just 32x32
        hidden_channels=256,
        num_heads=8,
        dropout=0.1,
        num_layers=12,
        num_classes=1000,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.cls_tok = nn.Parameter(torch.randn(hidden_channels))
        self.sin_emb = SinusodialEmbedding(hidden_channels)
        self.linear = nn.Conv2d(input_channels * patch_size**2, hidden_channels, 1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_channels,
                num_heads,
                hidden_channels * 4,
                dropout,
                norm_first=False,
            ),
            num_layers=num_layers,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, 4 * hidden_channels),
            nn.GELU(),
            nn.Linear(4 * hidden_channels, num_classes),
        )

    def forward(self, x):
        assert x.shape[-1] % self.patch_size == 0
        assert x.shape[-2] % self.patch_size == 0
        i = j = self.patch_size
        x = rearrange(x, "b d (h i) (w j) -> b (d i j) h w", i=i, j=j)
        x = self.linear(x)  # b d h w
        # Here we use 2d pe
        x = self.sin_emb.add_pe_2d(x)
        x = rearrange(x, "b d h w -> (h w) b d")
        # And don't add pe to cls tok
        e = repeat(self.cls_tok, "d -> 1 b d", b=x.shape[1])
        x = torch.cat([e, x], dim=0)
        x = self.transformer(x)
        x = self.mlp(x[0])  # t b d -> b d
        return x
