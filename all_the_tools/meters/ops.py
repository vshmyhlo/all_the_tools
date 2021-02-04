import numpy as np
import torch


def ones_like(x):
    if isinstance(x, torch.Tensor):
        return torch.ones_like(x)
    else:
        return np.ones_like(x)


def reduce_sum(x, dim):
    if isinstance(x, torch.Tensor):
        if dim is None:
            dim = ()
        return x.sum(dim=dim)
    else:
        return np.sum(x, axis=dim)


def concat(x, dim):
    assert isinstance(x, (list, tuple))
    if isinstance(x[0], torch.Tensor):
        return torch.cat(x, dim=dim)
    else:
        return np.concatenate(x, axis=dim)


def stack(x, dim):
    assert isinstance(x, (list, tuple))
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x, dim=dim)
    else:
        return np.stack(x, axis=dim)
