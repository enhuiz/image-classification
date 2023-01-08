from torchvision.models import resnet18, resnet34, resnet50

from .resnet_utils import update_resnet_
from .toy import ToyDilatedModel
from .vit import ViT


def get_model():
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

    return model
