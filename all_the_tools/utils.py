def weighted_sum(a, b, w):
    assert 0 <= w <= 1
    return w * a + (1 - w) * b
