from torch import nn
from torchvision.models import ResNet, resnet18, resnet34, resnet50

from .resnet_utils import update_resnet_
from .toy import ToyDilatedModel
from .vit import ViT
from .vit_aux import ViTAux
from .vit_native import ViTN


def get_model(num_classes=None):
    from ..config import cfg

    if cfg.model.lower() == "resnet18":
        model = resnet18()
    elif cfg.model.lower().lower() == "resnet34":
        model = resnet34()
    elif cfg.model.lower() == "resnet50":
        model = resnet50()
    elif cfg.model.lower() == "toy-dilated":
        model = ToyDilatedModel()
    elif cfg.model.lower() == "vit-12x256-8x8":
        model = ViT(
            num_layers=12,
            hidden_channels=256,
            num_heads=4,
            patch_size=8,
        )
    elif cfg.model.lower() == "vitn-12x256-8x8":
        model = ViTN(
            num_layers=12,
            hidden_channels=256,
            num_heads=4,
            patch_size=8,
        )
    elif cfg.model.lower() == "vitn-12x256-8x8-ra":
        model = ViTN(
            num_layers=12,
            hidden_channels=256,
            num_heads=4,
            patch_size=8,
            rel_attn=True,
        )
    elif cfg.model.lower() == "vitn-12x256-8x8-dm":
        model = ViTN(
            num_layers=12,
            hidden_channels=256,
            num_heads=4,
            patch_size=8,
            diag_mask=True,
        )
    elif cfg.model.lower() == "vitn-12x256-8x8-lt":
        model = ViTN(
            num_layers=12,
            hidden_channels=256,
            num_heads=4,
            patch_size=8,
            τ_type="log",
        )
    elif cfg.model.lower() == "vitn-12x256-8x8-ltv":
        model = ViTN(
            num_layers=12,
            hidden_channels=256,
            num_heads=4,
            patch_size=8,
            τ_type="vanilla",
        )
    elif cfg.model.lower() == "vit-aux-3x4x256-8x8":
        model = ViTAux(
            num_layers_per_block=4,
            num_blocks=3,
            hidden_channels=256,
            num_heads=4,
            patch_size=8,
        )
    elif cfg.model.lower() == "vit-24x256-8x8":
        model = ViT(
            num_layers=24,
            hidden_channels=256,
            num_heads=4,
            patch_size=8,
        )
    elif cfg.model.lower() == "vit-b-8x8":
        model = ViT(
            num_layers=12,
            hidden_channels=768,
            num_heads=12,
            patch_size=8,
        )
    else:
        raise NotImplementedError(cfg.model.lower())

    if cfg.model.lower().startswith("resnet"):
        update_resnet_(model)
        assert isinstance(model, ResNet)
        if num_classes is not None:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
