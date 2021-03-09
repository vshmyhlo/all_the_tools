import torch

from all_the_tools.torch.nn import TLU, FilterResponseNorm


def test_filter_response_norm():
    m = FilterResponseNorm(8)
    input = torch.zeros(4, 8, 10, 10)
    output = m(input)
    assert output.size() == (4, 8, 10, 10)


def test_tlu():
    m = TLU(8)
    input = torch.zeros(4, 8, 10, 10)
    output = m(input)
    assert output.size() == (4, 8, 10, 10)
