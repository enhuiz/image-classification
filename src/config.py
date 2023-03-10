from dataclasses import dataclass
from pathlib import Path

from .utils.config import Config as ConfigBase


@dataclass(frozen=True)
class Config(ConfigBase):
    data_dir: Path = Path("data")
    dataset: str = "cifar100"
    model: str = "resnet18"

    batch_size: int = 128
    eval_batch_size: int = 128
    warmup_min_lr: float = 1e-6
    warmup_max_lr: float = 2e-4
    warmup_num_steps: int = 500
    max_iter: int = 5_000
    gradient_clipping: float = 100
    eval_every: int = 100
    save_every: int = 5_000

    finetune_level: int = 5
    mixstyle_level: int = 0
    resnet_norm_type: str = "bn"
    normalizer: bool = True
    per_channel_normalizer: bool = False
    tailor_num_classes: bool = False
    constant_lr: float | None = None
    use_augmentation: bool = False

    @property
    def num_classes(self):
        if self.dataset.lower() in ["mnist", "cifar10"]:
            return 10
        elif self.dataset.lower() in ["cifar100"]:
            return 100
        else:
            raise NotImplementedError(self.dataset)

    @property
    def warmup_decay_scheduler(self):
        return {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": self.warmup_min_lr,
                "warmup_max_lr": self.warmup_max_lr,
                "warmup_num_steps": self.warmup_num_steps,
                "total_num_steps": self.max_iter,
                "warmup_type": "linear",
            },
        }

    @property
    def constant_scheduler(self):
        return {
            "type": "WarmupLR",
            "params": {
                # Keep the same
                "warmup_min_lr": self.constant_lr,
                "warmup_max_lr": self.constant_lr,
                "warmup_num_steps": 1,
            },
        }

    @property
    def scheduler(self):
        if self.constant_lr is not None:
            return self.constant_scheduler
        else:
            return self.warmup_decay_scheduler

    @property
    def ds_cfg(self):
        return {
            "train_micro_batch_size_per_gpu": self.batch_size,
            "gradient_accumulation_steps": 1,
            "optimizer": {"type": "Adam"},
            "scheduler": self.scheduler,
            "gradient_clipping": self.gradient_clipping,
        }


cfg = Config.from_cli()

if __name__ == "__main__":
    print(cfg)
