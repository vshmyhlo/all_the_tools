import numpy as np
import pytest
import torch

from all_the_tools.meters.concat import Concat

p = [
    (lambda x: x, lambda x: x),
    (np.array, lambda x: x),
    (torch.tensor, lambda x: x.data.cpu().numpy()),
]


@pytest.mark.parametrize("to_t,to_a", p)
def test_concat(to_t, to_a):
    # scalar
    m = Concat()
    m.update(to_t(0))
    m.update(to_t(1))
    with pytest.raises((ValueError, RuntimeError)):
        m.compute_and_reset()

    # array
    m = Concat()
    m.update(to_t([[0, 1, 2]]))
    m.update(to_t([[1, 3, 5]]))
    expected = [
        [0, 1, 2],
        [1, 3, 5],
    ]
    assert np.array_equal(to_a(m.compute_and_reset()), expected)

    # array over dim
    m = Concat(1)
    m.update(to_t([[0, 1, 2]]))
    m.update(to_t([[1, 3, 5]]))
    expected = [
        [0, 1, 2, 1, 3, 5],
    ]
    assert np.array_equal(to_a(m.compute_and_reset()), expected)
