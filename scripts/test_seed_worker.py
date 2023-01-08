import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def _seed_worker(worker_id):
    print(torch.initial_seed())
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


dl = DataLoader(
    [*range(10)],
    batch_size=2,
    shuffle=True,
    worker_init_fn=_seed_worker,
    num_workers=2,
    drop_last=True,
)

print([x for x in dl])
print([x for x in dl])
