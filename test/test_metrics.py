import numpy as np

from all_the_tools.metrics import Mean


def test_mean():
    m = Mean()

    m.update([0, 1, 2, 3, 4])
    m.update([0, 1, 2, 3, 4])

    assert np.array_equal(m.compute_and_reset(), 2)

    m = Mean(0)
    m.update([[0, 1, 2]] * 2)
    m.update([[0, 1, 2]] * 3)

    assert np.array_equal(m.compute_and_reset(), [0, 1, 2])
