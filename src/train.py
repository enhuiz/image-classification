import json
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from tqdm import tqdm

from .config import cfg
from .models import get_model
from .normalizer import Normalizer
from .utils import Diagnostic, setup_logging, to_device, trainer

_logger = logging.getLogger(__name__)


def load_engines():
    model = get_model(cfg.num_classes if cfg.tailor_num_classes else None)

    if cfg.normalizer:
        if cfg.per_channel_normalizer:
            model.normalizer = Normalizer(num_channels=3, axis=1)
        else:
            model.normalizer = Normalizer()

    engines = dict(
        model=trainer.Engine(
            model=model,
            config=cfg.ds_cfg,
        ),
    )

    return trainer.load_engines(engines, cfg)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    setup_logging(cfg.log_dir)

    if cfg.dataset.lower() == "mnist":
        Dataset = MNIST
    elif cfg.dataset.lower() == "cifar10":
        Dataset = CIFAR10
    elif cfg.dataset.lower() == "cifar100":
        Dataset = CIFAR100
    else:
        raise NotImplementedError(cfg.dataset)

    train_ds = Dataset(
        str(cfg.data_dir),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    test_ds = Dataset(
        str(cfg.data_dir),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        worker_init_fn=_seed_worker,
    )

    train_200_dl = DataLoader(
        Subset(train_ds, [*range(200)]),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        drop_last=False,
        worker_init_fn=_seed_worker,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        drop_last=False,
        worker_init_fn=_seed_worker,
    )

    del train_ds, test_ds

    diagnostic = None

    def train_feeder(engines, batch, name):
        nonlocal diagnostic
        del name

        x, y = batch

        model = engines["model"]

        if diagnostic is None:
            diagnostic = Diagnostic(model.module)

        if trainer.get_cmd() == "diag-stop":
            diagnostic.save()

        if trainer.get_cmd() == "diag-start":
            diagnostic.attach()

        if cfg.normalizer:
            x_normed = model.normalizer(x)
        else:
            x_normed = x

        if "aux" in cfg.model:
            logits = model(x_normed, y)
        else:
            logits = model(x_normed)

        ce_loss = F.cross_entropy(logits, y)
        aux_losses = engines.gather_attribute("loss")
        loss = ce_loss + sum(aux_losses.values())

        top1_acc = (logits.argmax(dim=-1) == y).float().mean().item()

        stats = {"loss.ce": ce_loss.item(), "acc.top1": top1_acc}
        stats |= {k: v.item() for k, v in aux_losses.items()}
        stats |= engines.gather_attribute("scalar")

        return loss, stats

    @torch.inference_mode()
    def run_eval(engines, name, dl):
        model = engines["model"]

        top1_accs = []

        for batch in tqdm(dl):
            batch = to_device(batch, cfg.device)
            x, y = batch

            if cfg.normalizer:
                x_normed = model.normalizer(x)
            else:
                x_normed = x

            logits = model(x_normed)

            top1_accs.extend((logits.argmax(dim=-1) == y).float())

        top1_acc = torch.stack(top1_accs).mean().item()

        log_dir = cfg.log_dir / str(engines.global_step) / name
        log_dir.mkdir(parents=True, exist_ok=True)

        with open(log_dir / "top1_acc.txt", "w") as f:
            f.write(str(top1_acc))

        stats = {}
        stats["global_step"] = engines.global_step
        stats["name"] = name
        stats["acc.top1"] = top1_acc

        _logger.info(f"{json.dumps(stats)}.")

    def eval_fn(engines):
        run_eval(engines, "train_200", train_200_dl)
        run_eval(engines, "test", test_dl)

    trainer.train(
        engines_loader=load_engines,
        train_dl=train_dl,
        train_feeder=train_feeder,
        eval_fn=eval_fn,
    )


if __name__ == "__main__":
    main()
