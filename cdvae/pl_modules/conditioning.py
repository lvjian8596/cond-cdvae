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

    def forward(self, prop):
        atom_types, num_atoms = prop
        batch = torch.repeat_interleave(
            torch.arange(num_atoms.size(0), device=num_atoms.device),
            num_atoms,
        )
        atom_emb = self.emb(atom_types)
        comp_emb = scatter(atom_emb, batch, dim=0, reduce=self.reduce)
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

    def forward(self, prop):
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


# ## [cG-SchNet](
# ## MISC
class GaussianExpansion(nn.Module):
    r"""Expansion layer using a set of Gaussian functions.

    https://github.com/atomistic-machine-learning/cG-SchNet/blob/53d73830f9fb1158296f060c2f82be375e2bb7f9/nn_classes.py#L687)

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

    def __init__(self, start, stop, n_gaussians=50, trainable=False, width=None):
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
    """Concatenate multi-embedding vector
    all sublayer should have a attribute named 'n_out'

        feat1 -> sub_layer1 \
                            concat -> MLP -> out
        feat2 -> sub_layer2 /

    Returns: z(B, out_dim)
    """

    def __init__(
        self,
        cond_keys: list,
        hidden_dim: int,
        fc_num_layers: int,
        cond_dim: int,
        types: dict,
        *args,
        **kwargs,
    ):
        """Concatenate multi-embedding vector

        Args:
            cond_keys (list): list of condition name strings
            hidden_dim (int): hidden dimensions of out MLP
            fc_num_layers (int): number of layers of out MLP
            out_dim (int): out dimension of MLP
            types (dict or dict-like): kwargs of sub-embedding modules
        """
        super().__init__()
        self.cond_keys = cond_keys

        n_in = 0
        self.sub_emb_list = nn.ModuleList()
        for cond_key in cond_keys:
            sub_emb = hydra.utils.instantiate(types[cond_key])
            self.sub_emb_list.append(sub_emb)
            n_in += sub_emb.n_out
        self.cond_mlp = build_mlp(n_in, hidden_dim, fc_num_layers, cond_dim)

    def forward(self, conditions: dict):
        # conditions={'composition': (atom_types, num_atoms), 'cond_name': cond_vals}
        cond_vecs = []
        for cond_key, sub_emb in zip(self.cond_keys, self.sub_emb_list):
            cond_vec = sub_emb(conditions[cond_key])
            cond_vecs += [cond_vec]

        cond_vecs = torch.cat(cond_vecs, dim=-1)
        cond_vec = self.cond_mlp(cond_vecs)
        return cond_vec


class AtomwiseConditioning(nn.Module):
    def __init__(self, cond_dim, atom_emb_size, mode: str = 'concat'):
        """Aggregate condition vector c with atomtype embedding vector

        Args:
            cond_dim (int): Dimension of condition vector
            atom_emb_size (int): Dimension of atom type embedding
            mode (str, optional): Aggregate mode. Defaults to 'concat'.
        """
        super().__init__()
        self.atom_emb = AtomEmbedding(atom_emb_size)
        self.agg = AggregateConditioning(cond_dim, atom_emb_size, mode)

    def forward(self, c, atom_types, num_atoms):
        atom_emb = self.atom_emb(atom_types)
        c_per_atom = c.repeat_interleave(num_atoms, dim=0)
        emb = self.agg(c_per_atom, atom_emb)
        return emb


class AggregateConditioning(nn.Module):
    def __init__(self, cond_dim: int, emb_dim: int, mode: str = 'concat'):
        """Aggregate condition vector c with embedding vector z, output z',
        always output the same dimension as embedding vector z

        Args:
            cond_dim (int): Dimension of condition vector, c_dim
            emb_dim (int): Dimension of input embedding vector's dim, z_dim
            mode (str, optional): Aggregate mode. ['concatenate', 'bias',
            'scale', 'film'] Defaults to 'concat'.
        """
        super().__init__()
        # TODO: change concat to pure concat
        if mode.startswith('concat') or mode.startswith('cat'):
            self.cond_model = ConcatConditioning(emb_dim, cond_dim)
        elif mode.startswith('bias'):
            self.cond_model = BiasConditioning(emb_dim, cond_dim)
        elif mode.startswith('scal'):
            self.cond_model = ScaleConditioning(emb_dim, cond_dim)
        elif mode.startswith('film'):
            self.cond_model = FiLM(emb_dim, cond_dim)
        else:
            raise ValueError("Unknown mode")

    def forward(self, c, z):  # return cond_z
        z = self.cond_model(z, c)
        return z


class ConcatConditioning(nn.Module):
    """z = Lin(cat[x, y])"""

    def __init__(self, xdim, ydim, zdim=None):
        super(ConcatConditioning, self).__init__()

        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim if zdim is not None else xdim

        self.linear = nn.Linear(self.xdim + self.ydim, self.zdim)

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
