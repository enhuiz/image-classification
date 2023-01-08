from torchvision.models import resnet18, resnet34, resnet50

from .resnet_utils import update_resnet_


def get_model():
    from ..config import cfg

    if cfg.model.lower() == "resnet18":
        ret = resnet18()
    elif cfg.model.lower().lower() == "resnet34":
        ret = resnet34()
    elif cfg.model.lower() == "resnet50":
        ret = resnet50()
    else:
        raise NotImplementedError(cfg.model.lower())

    if cfg.model.lower().startswith("resnet"):
        update_resnet_(ret)

    return ret
