from torchvision.models import resnet18, resnet34, resnet50


def get_model():
    from ..config import cfg

    if cfg.model is "resnet18":
        ret = resnet18()
    elif cfg.model is "resnet34":
        ret = resnet34()
    elif cfg.model is "resnet50":
        ret = resnet50()
    else:
        raise NotImplementedError(cfg.model)

    return ret
