"""x|y"""
import torch
import torch.nn as nn


class ConcatConditioning(nn.Module):
    """z = Lin(cat[x, y])"""

    def __init__(self, xdim, ydim, zdim):
        super(ConcatConditioning, self).__init__()

        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim

        self.linear = nn.Linear(xdim + ydim, zdim)

    def forward(self, x, y):
        assert x.size(1) == self.xdim
        assert y.size(1) == self.ydim

        m = torch.cat([x, y], axis=1)
        z = self.linear(m)
        return z


class BiasConditioning(nn.Module):
    """z = x + Lin(y)"""

    def __init__(self, xdim, ydim):
        super(BiasConditioning, self).__init__()

        self.xdim = xdim
        self.ydim = ydim

        self.linear = nn.Linear(ydim, xdim)

    def forward(self, x, y):
        assert x.size(1) == self.xdim
        assert y.size(1) == self.ydim

        z = x + self.linear(y)
        return z


class ScaleConditioning(nn.Module):
    """z = x * Lin(y)"""

    def __init__(self, xdim, ydim):
        super(ScaleConditioning, self).__init__()

        self.xdim = xdim
        self.ydim = ydim

        self.linear = nn.Linear(ydim, xdim)

    def forward(self, x, y):
        assert x.size(1) == self.xdim
        assert y.size(1) == self.ydim

        z = x * self.linear(y)
        return z


class FiLM(nn.Module):
    """z = γ(y) * x + β(y)"""

    def __init__(self, xdim, ydim):
        super(FiLM, self).__init__()
        self.xdim = xdim
        self.ydim = ydim
        self.gamma = nn.Linear(ydim, xdim)
        self.beta = nn.Linear(ydim, xdim)

    def forward(self, x, y):
        assert x.size(1) == self.xdim
        assert y.size(1) == self.ydim

        z = self.gamma(y) * x + self.beta(y)
        return z


if __name__ == '__main__':
    bs, xdim, ydim = 16, 32, 6
    film_cond = FiLM(xdim, ydim)
    x = torch.rand(bs, xdim)
    y = torch.rand(bs, ydim)
    z = film_cond(x, y)
    print('x shape : ', x.shape)
    print('y shape : ', y.shape)
    print('z shape : ', z.shape)
