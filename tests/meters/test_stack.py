import numpy as np
import pytest
import torch

from all_the_tools.meters.stack import Stack

p = [
    (lambda x: x, lambda x: x),
    (np.array, lambda x: x),
    (torch.tensor, lambda x: x.data.cpu().numpy()),
]


@pytest.mark.parametrize("to_t,to_a", p)
def test_stack(to_t, to_a):
    # scalar
    m = Stack()
    m.update(to_t(0))
    m.update(to_t(1))
    assert np.array_equal(to_a(m.compute_and_reset()), [0, 1])

    # array
    m = Stack()
    m.update(to_t([[0, 1, 2]]))
    m.update(to_t([[1, 3, 5]]))
    expected = [
        [[0, 1, 2]],
        [[1, 3, 5]],
    ]
    assert np.array_equal(to_a(m.compute_and_reset()), expected)

    # array over dim
    m = Stack(1)
    m.update(to_t([[0, 1, 2]]))
    m.update(to_t([[1, 3, 5]]))
    expected = [
        [[0, 1, 2], [1, 3, 5]],
    ]
    assert np.array_equal(to_a(m.compute_and_reset()), expected)
