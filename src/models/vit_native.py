import torch
from einops import rearrange, repeat
from torch import Tensor, einsum, nn

from .common import SinusodialEmbedding


class Attention(nn.Module):
    """
    The idea of diag_mask and learnable_temp comes from: https://arxiv.org/pdf/2112.13492.pdf
    Rel pos comes form: https://arxiv.org/pdf/1901.02860.pdf
    """

    def __init__(self, num_channels, num_heads, rel_attn, diag_mask, τ_type):
        super().__init__()
        assert num_channels % num_heads == 0
        dim_head = num_channels // num_heads
        self.num_heads = num_heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(num_channels, num_channels * 3, bias=False)
        self.to_out = nn.Linear(num_channels, num_channels)

        if rel_attn:
            self.to_pos = nn.Linear(num_channels, num_channels, bias=False)
            self.pos_u = nn.parameter.Parameter(torch.randn(num_heads, dim_head))
            self.pos_v = nn.parameter.Parameter(torch.randn(num_heads, dim_head))
            self.pos_emb = SinusodialEmbedding(num_channels)

        if τ_type == "log":
            # Instead of simply use a learnable temp τ
            # Here I learn log 1/τ, which avoid division and sign flipping
            self.log_one_div_by_τ = nn.parameter.Parameter(torch.zeros(1))
        elif τ_type == "vanilla":
            self.τ = nn.parameter.Parameter(torch.ones(1))
        else:
            assert τ_type is None

        self.rel_attn = rel_attn
        self.diag_mask = diag_mask
        self.τ_type = τ_type

    def forward(self, x):
        """
        Args:
            x: (t b d)
        Returns:
            x: (t b d)
        """
        device = x.device
        h = self.num_heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "t b (h d) -> t b h d", h=h), (q, k, v))

        if self.rel_attn:
            t = len(q)
            seq = torch.arange(t, device=device)
            rel_dist = rearrange(seq, "i -> i 1") - rearrange(seq, "j -> 1 j")
            rpe = self.to_pos(self.pos_emb(rel_dist))  # (i j d)
            rpe = repeat(rpe, "i j (h d) -> i j h d", h=h)

            energy = sum(
                [
                    # https://arxiv.org/pdf/1901.02860.pdf, a b c d
                    einsum("i b h d, j b h d -> i j b h", q, k),
                    einsum("i b h d, i j h d -> i j b h", q, rpe),
                    einsum("h d, i j b h d -> i j b h", self.pos_u, k.unsqueeze(0)),
                    einsum("h d, i j h d -> i j h", self.pos_v, rpe).unsqueeze(2),
                ]
            )

            assert isinstance(energy, Tensor)
        else:
            energy = einsum("i b h d, j b h d -> i j b h", q, k)

        energy = energy * self.scale

        if self.diag_mask:
            assert energy.shape[0] == energy.shape[1]  # self attention
            mask = torch.eye(energy.shape[0], device=device).bool()
            mask = mask.unsqueeze(-1).unsqueeze(-1)  # (i j 1 1)
            energy.masked_fill_(mask, -torch.finfo(energy.dtype).max)

        if self.τ_type == "log":
            energy = energy * self.log_one_div_by_τ.exp()
        elif self.τ_type == "vanilla":
            energy = energy / self.τ
        else:
            assert self.τ_type is None

        attn = energy.softmax(dim=1)  # (i j b h)

        out = einsum("i j b h, j b h d -> i b h d", attn, v)
        out = rearrange(out, "i b h d -> i b (h d)")
        out = self.to_out(out)

        return out


class PrenormResidual(nn.Module):
    def __init__(self, block, num_channels, dropout):
        super().__init__()
        self.block = block
        self.norm = nn.LayerNorm(num_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.block(self.norm(x)))


class TransformerEncoderLayer(nn.Sequential):
    def __init__(
        self,
        num_channels,
        num_heads,
        dropout,
        rel_attn,
        diag_mask,
        τ_type,
    ):
        super().__init__(
            PrenormResidual(
                Attention(num_channels, num_heads, rel_attn, diag_mask, τ_type),
                num_channels=num_channels,
                dropout=dropout,
            ),
            PrenormResidual(
                nn.Sequential(
                    nn.Linear(num_channels, num_channels * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(num_channels * 4, num_channels),
                ),
                num_channels=num_channels,
                dropout=dropout,
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(
        self,
        num_channels,
        num_heads,
        dropout,
        num_layers,
        rel_attn,
        diag_mask,
        τ_type,
    ):
        super().__init__(
            *[
                TransformerEncoderLayer(
                    num_channels,
                    num_heads,
                    dropout,
                    rel_attn,
                    diag_mask,
                    τ_type,
                )
                for _ in range(num_layers)
            ]
        )


class ViTN(nn.Module):
    def __init__(
        self,
        input_channels=3,
        patch_size=8,  # not using 16x16 as our image is just 32x32
        hidden_channels=256,
        num_heads=8,
        dropout=0.1,
        num_layers=12,
        num_classes=1000,
        diag_mask=False,
        rel_attn=False,
        τ_type=None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.cls_tok = nn.Parameter(torch.randn(hidden_channels))
        self.sin_emb = SinusodialEmbedding(hidden_channels)
        self.linear = nn.Conv2d(input_channels * patch_size**2, hidden_channels, 1)
        self.transformer = TransformerEncoder(
            num_channels=hidden_channels,
            diag_mask=diag_mask,
            rel_attn=rel_attn,
            τ_type=τ_type,
            dropout=dropout,
            num_heads=num_heads,
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
