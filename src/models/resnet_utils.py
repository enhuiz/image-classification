import logging
import re

import torch
from torch import Tensor, nn

from ..config import cfg
from .fbn import FrozenBatchNorm2d
from .mix_style import MixStyle

_logger = logging.getLogger(__name__)


def _setattr_recur(model, name, value):
    if "." in name:
        head, tails = name.split(".", 1)
        _setattr_recur(getattr(model, head), tails, value)
    else:
        setattr(model, name, value)


def _replace_norm(model, pattern, num_groups=16):
    for name, bn in list(model.named_modules()):
        if isinstance(bn, nn.BatchNorm2d) and re.match(pattern, name):
            if cfg.resnet_norm_type == "gn":
                gn = nn.GroupNorm(num_groups, bn.num_features)
                _logger.info(f"Setting {name} {bn} to {gn}")
                _setattr_recur(model, name, gn)


def _freeze_bn(model, pattern):
    for name, bn in list(model.named_modules()):
        if isinstance(bn, nn.BatchNorm2d) and re.match(pattern, name):
            fbn = FrozenBatchNorm2d(bn.num_features)
            fbn.load_state_dict(bn.state_dict())
            _logger.info(f"Setting {name} {bn} to {fbn}")
            _setattr_recur(model, name, fbn)


def _freeze(model):
    for name, p in model.named_parameters():
        _logger.info(f"{name} is frozen.")
        p.requires_grad_(False)


def update_resnet_(model):
    if cfg.finetune_level >= 5:
        _replace_norm(model, "^bn1")
    else:
        _freeze_bn(model, "^bn1")
        _freeze(model.conv1)

    if cfg.finetune_level >= 4:
        _replace_norm(model, "layer1")
    else:
        _freeze_bn(model, "layer1")
        _freeze(model.layer1)

    if cfg.finetune_level >= 3:
        _replace_norm(model, "layer2")
    else:
        _freeze_bn(model, "layer2")
        _freeze(model.layer2)

    if cfg.finetune_level >= 2:
        _replace_norm(model, "layer3")
    else:
        _freeze_bn(model, "layer3")
        _freeze(model.layer3)

    if cfg.finetune_level >= 1:
        _replace_norm(model, "layer4")
    else:
        _freeze_bn(model, "layer4")
        _freeze(model.layer4)

    if cfg.mixstyle_level > 0:
        model.mix_style = MixStyle()

    def _forward_impl(x: Tensor, self=model) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        if cfg.mixstyle_level >= 1 and x.requires_grad:
            x = self.mix_style(x)

        x = self.layer2(x)

        if cfg.mixstyle_level >= 2 and x.requires_grad:
            x = self.mix_style(x)

        x = self.layer3(x)

        if cfg.mixstyle_level >= 3 and x.requires_grad:
            x = self.mix_style(x)

        x = self.layer4(x)

        if cfg.mixstyle_level >= 4 and x.requires_grad:
            x = self.mix_style(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    model._forward_impl = _forward_impl

    return model
