import torch
import torch.nn as nn


class FilterResponseNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
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
