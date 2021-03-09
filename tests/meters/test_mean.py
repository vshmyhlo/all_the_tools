import numpy as np
import pytest
import torch

from all_the_tools.meters.mean import Mean

p = [
    (lambda x: x, lambda x: x),
    (np.array, lambda x: x),
    (torch.tensor, lambda x: x.data.cpu().numpy()),
]


@pytest.mark.parametrize("to_t,to_a", p)
def test_mean(to_t, to_a):
    # scalar
    m = Mean()
    m.update(to_t(0))
    m.update(to_t(1))
    assert np.array_equal(to_a(m.compute_and_reset()), 0.5)

    # array
    m = Mean()
    m.update(to_t([[0, 1, 2]]))
    m.update(to_t([[1, 3, 5]]))
    print(m.compute())
    assert np.array_equal(to_a(m.compute_and_reset()), 2.0)

    # array over dim
    m = Mean(0)
    m.update(to_t([[0, 1, 2]]))
    m.update(to_t([[1, 3, 5]]))
    assert np.array_equal(to_a(m.compute_and_reset()), [0.5, 2, 3.5])
