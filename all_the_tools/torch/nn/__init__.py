import torch
import torch.nn as nn


class FilterResponseNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.eps_p = nn.Parameter(torch.empty(1, num_features, 1, 1))
        self.weight = nn.Parameter(torch.empty(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.empty(1, num_features, 1, 1))

        nn.init.zeros_(self.eps_p)
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        nu = (input**2).mean((2, 3), keepdim=True)
        eps = self.eps + self.eps_p.abs()
        nu = torch.sqrt(nu + eps)

        input = input / nu
        input = input * self.weight + self.bias

        return input


class TLU(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.tau = nn.Parameter(torch.empty(1, num_features, 1, 1))

    def forward(self, input):
        input = torch.max(input, self.tau)

        return input


class PositionalEncoding(nn.Module):
    def forward(self, input, start=0):
        _, t, c = input.size()

        pos = start + torch.arange(t, dtype=input.dtype, device=input.device).unsqueeze(1)
        i = torch.arange(c, dtype=input.dtype, device=input.device).unsqueeze(0)
        enc = pos / 10000**(2 * i / c)
        enc = torch.cat([
            torch.sin(enc[:, 0::2]),
            torch.cos(enc[:, 1::2]),
        ], 1)
        enc = enc.unsqueeze(0)

        input = input + enc

        return input
