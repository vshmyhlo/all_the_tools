from all_the_tools.meters.last import Last


def test_last():
    m = Last()
    assert m.compute_and_reset() is None
    m.update(object())
    o = object()
    m.update(o)
    assert m.compute_and_reset() is o
