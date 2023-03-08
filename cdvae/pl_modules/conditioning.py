"""p(x|y)"""
import hydra
import torch
import torch.nn as nn

from torch_scatter import scatter
from cdvae.pl_modules.basic_blocks import build_mlp
from cdvae.pl_modules.embeddings import KHOT_EMBEDDINGS, MAX_ATOMIC_NUM
from cdvae.pl_modules.gemnet.layers.embedding_block import AtomEmbedding


class SubEmbedding(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.n_out = n_out


class CompositionEmbedding(SubEmbedding):
    def __init__(self, n_out, reduce='sum'):
        super().__init__(n_out)
        self.n_out = n_out
        self.reduce = reduce
        self.emb = AtomEmbedding(n_out)

    def forward(self, batch):
        atom_emb = self.emb(batch.atom_types)
        comp_emb = scatter(atom_emb, batch.batch, dim=0, reduce=self.reduce)
        return comp_emb


# single scalar, one sample is (1,)
class ScalarEmbedding(SubEmbedding):
    def __init__(
        self,
        prop_name: str,
        hidden_dim: int,
        fc_num_layers: int,
        n_out: int,
        batch_norm: bool,
        # gaussian expansion
        n_basis: int,  # num gaussian basis
        start: float = None,
        stop: float = None,
        trainable_gaussians: bool = False,
        width: float = None,
        no_expansion: bool = False,
    ):
        super().__init__(n_out)
        self.n_out = n_out
        self.prop_name = prop_name
        if batch_norm:
            self.bn = nn.BatchNorm1d(1)
        else:
            self.bn = None
        if not no_expansion:
            self.expansion_net = GaussianExpansion(
                start, stop, n_basis, trainable_gaussians, width
            )
            self.mlp = build_mlp(n_basis, hidden_dim, fc_num_layers, n_out)
        else:
            self.expansion_net = None
            self.mlp = build_mlp(1, hidden_dim, fc_num_layers, n_out)

    def forward(self, batch):
        prop = batch[self.prop_name]
        if self.bn is not None:
            prop = self.bn(prop)
        if self.expansion_net is not None:
            prop = self.expansion_net(prop)  # expanded prop
        out = self.mlp(prop)
        return out


class VectorEmbedding(SubEmbedding):
    def __init__(self, prop_name, n_in, hidden_dim, fc_num_layers, n_out):
        super().__init__(n_out)
        self.prop_name = prop_name
        self.mlp = build_mlp(n_in, hidden_dim, fc_num_layers, n_out)

    def forward(self, batch):
        prop = batch[self.prop_name]
        return self.mlp(prop)


### [cG-SchNet](https://github.com/atomistic-machine-learning/cG-SchNet/blob/53d73830f9fb1158296f060c2f82be375e2bb7f9/nn_classes.py#L687)
### MISC
class GaussianExpansion(nn.Module):
    r"""Expansion layer using a set of Gaussian functions.
    Args:
        start (float): center of first Gaussian function, :math:`\mu_0`.
        stop (float): center of last Gaussian function, :math:`\mu_{N_g}`.
        n_gaussians (int, optional): total number of Gaussian functions, :math:`N_g`
            (default: 50).
        trainable (bool, optional): if True, widths and offset of Gaussian functions
            are adjusted during training process (default: False).
        widths (float, optional): width value of Gaussian functions (provide None to
            set the width to the distance between two centers :math:`\mu`, default:
            None).
    """

    def __init__(
        self, start, stop, n_gaussians=50, trainable=False, width=None
    ):
        super(GaussianExpansion, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        if width is None:
            widths = torch.FloatTensor(
                (offset[1] - offset[0]) * torch.ones_like(offset)
            )
        else:
            widths = torch.FloatTensor(width * torch.ones_like(offset))
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, property):
        """Compute expanded gaussian property values.
        Args:
            property (torch.Tensor): property values of (N_b x 1) shape.
        Returns:
            torch.Tensor: layer output of (N_b x N_g) shape.
        """
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(self.widths, 2)[None, :]
        # Use advanced indexing to compute the individual components
        diff = property - self.offsets[None, :]
        # compute expanded property values
        return torch.exp(coeff * torch.pow(diff, 2))


class MultiEmbedding(nn.Module):
    """
    feat1 -> sub_layer1 \
                        concat -> MLP -> out
    feat2 -> sub_layer2 /

    all sublayer should have a attribute named 'n_out'
    """

    def __init__(
        self,
        cond_names,
        hidden_dim,  # out MLP
        fc_num_layers: int,  # out MLP
        out_dim: int,  # out MLP
        types,  # kwargs of sub-embedding models
    ):
        super().__init__()
        self.cond_names = cond_names

        n_in = 0
        self.sub_emb_list = []
        for cond_name in cond_names:
            sub_emb = hydra.utils.instantiate(types[cond_name])
            self.sub_emb_list.append(sub_emb)
            n_in += sub_emb.n_out
        self.cond_mlp = build_mlp(n_in, hidden_dim, fc_num_layers, out_dim)

    def forward(self, batch):
        cond_vecs = []
        for cond_name, sub_emb in zip(self.cond_names, self.sub_emb_list):
            cond_vec = sub_emb(batch)
            cond_vecs += [cond_vec]

        cond_vecs = torch.cat(cond_vecs, dim=-1)
        cond_vec = self.cond_mlp(cond_vecs)
        return cond_vec


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
