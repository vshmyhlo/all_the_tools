import random

import numpy as np
import torch
import torch.nn as nn

from all_the_tools.utils import weighted_sum


class ModuleEMA:
    def __init__(self, ema: nn.Module, a: float):
        self.ema = ema
        self.a = a

    def update(self, new: nn.Module):
        for p_ema, p_new in zip(self.ema.parameters(), new.parameters()):
            p_ema.data = weighted_sum(p_ema.data, p_new.data, self.a)
        for b_ema, b_new in zip(self.ema.buffers(), new.buffers()):
            b_ema.data = b_new.data


class Saver(object):
    def __init__(self, objects):
        assert "epoch" not in objects

        self.objects = objects

    def save(self, path, epoch):
        state = {
            **{k: self.objects[k].state_dict() for k in self.objects},
            "epoch": epoch,
        }
        torch.save(state, path)

    def load(self, path, keys=None):
        if keys is None:
            keys = self.objects.keys()

        state = torch.load(path)
        for k in keys:
            self.objects[k].load_state_dict(state[k])

        return state["epoch"]


def one_hot(input, n, dtype=torch.float):
    return torch.eye(n, dtype=dtype, device=input.device)[input]


def random_seed(seed):
    # python
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def worker_init_fn(_):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)
