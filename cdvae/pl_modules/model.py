from typing import Any, Dict

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from cdvae.common.data_utils import (
    EPSILON,
    cart_to_frac_coords,
    frac_to_cart_coords,
    lengths_angles_to_volume,
    mard,
    min_distance_sqr_pbc,
    StandardScalerTorch,
)
from cdvae.common.utils import PROJECT_ROOT
from cdvae.pl_modules.embeddings import KHOT_EMBEDDINGS, MAX_ATOMIC_NUM
from cdvae.pl_modules.gemnet.layers.embedding_block import AtomEmbedding
from cdvae.pl_modules.conditioning import (
    ConcatConditioning,
    BiasConditioning,
    ScaleConditioning,
    FiLM,
)


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class CompositionCondition(nn.Module):
    """ z on condition of (c, N)
    z:                        (Batch, lattent_dim)
    c: atom_types, each atom  (Nnode,)
    N: num_atoms, each sample (Batch,)

    condition: Lin(Agg(emb(c), N))) -> composition embedding of each sample

    c -> Emb(c)              (Nnode, emb_size_atom)  atom embedding
    Emb(c) -> Agg(Emb(c))    (Batch, emb_size_atom)  sample embedding

    1. Concatenate 2. Bias 3. Scale 4. FiLM
    """

    def __init__(self, emb_size_atom, latent_dim, mode='concatenate') -> None:
        super().__init__()
        self.mode = mode
        self.atom_emb = AtomEmbedding(emb_size_atom)

        if self.mode.startswith('concat') or self.mode.startswith('cat'):
            self.cond_model = ConcatConditioning(
                latent_dim, emb_size_atom, latent_dim
            )
        elif self.mode.startswith('bias'):
            self.cond_model = BiasConditioning(latent_dim, emb_size_atom)
        elif self.mode.startswith('scal'):
            self.cond_model = ScaleConditioning(latent_dim, emb_size_atom)
        elif self.mode.startswith('film'):
            self.cond_model = FiLM(latent_dim, emb_size_atom)
        else:
            raise ValueError("Unknown mode")

    def forward(self, z, atom_types, num_atoms):  # return cond_z
        batch = torch.arange(
            len(num_atoms), device=num_atoms.device
        ).repeat_interleave(num_atoms)
        # (Nnode, emb)
        atom_emb = self.atom_emb(atom_types)
        # aggregate  (Batch, emb)
        sample_emb = scatter(atom_emb, batch, dim=0, reduce='mean')

        z = self.cond_model(z, sample_emb)
        return z


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer,
            params=self.parameters(),
            _convert_="partial",
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 5,
                "monitor": "val_loss",
            },
        }


class CDVAE(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = hydra.utils.instantiate(
            self.hparams.encoder, num_targets=self.hparams.latent_dim
        )
        self.decoder = hydra.utils.instantiate(self.hparams.decoder)

        self.comp_cond = CompositionCondition(
            self.hparams.hidden_dim, self.hparams.latent_dim
        )

        self.fc_mu = nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim)
        self.fc_var = nn.Linear(
            self.hparams.latent_dim, self.hparams.latent_dim
        )

        self.fc_num_atoms = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            self.hparams.max_atoms + 1,
        )
        self.fc_lattice = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            6,
        )
        self.fc_composition = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            MAX_ATOMIC_NUM,
        )
        # for property prediction.
        if self.hparams.predict_property:
            self.fc_property = build_mlp(
                self.hparams.latent_dim,
                self.hparams.hidden_dim,
                self.hparams.fc_num_layers,
                1,
            )

        sigmas = torch.tensor(
            np.exp(
                np.linspace(
                    np.log(self.hparams.sigma_begin),
                    np.log(self.hparams.sigma_end),
                    self.hparams.num_noise_level,
                )
            ),
            dtype=torch.float32,
        )

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        type_sigmas = torch.tensor(
            np.exp(
                np.linspace(
                    np.log(self.hparams.type_sigma_begin),
                    np.log(self.hparams.type_sigma_end),
                    self.hparams.num_noise_level,
                )
            ),
            dtype=torch.float32,
        )

        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        self.embedding = torch.zeros(100, 92)
        for i in range(100):
            self.embedding[i] = torch.tensor(KHOT_EMBEDDINGS[i + 1])

        # obtain from datamodule.
        self.lattice_scaler = StandardScalerTorch(
            torch.tensor(0), torch.tensor(1)
        )
        self.scaler = StandardScalerTorch(torch.tensor(0), torch.tensor(1))

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        logvar = torch.clamp(logvar, None, 4)
        std = torch.exp(0.5 * logvar)
        assert torch.isfinite(logvar).all()
        assert torch.isfinite(std).all()
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch):
        """
        encode crystal structures to latents.
        """
        hidden = self.encoder(batch)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        # ======== debug =========
        # for parm in self.fc_var.parameters():
        #     if torch.max(parm) > 1000:
        #         print(torch.max(parm), torch.max(log_var).item())
        # if torch.max(hidden) > 1000:
        #     print(batch)
        #     for parm in self.fc_var.parameters():
        #         print(torch.max(parm))
        # ========================
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    def decode_stats(
        self,
        z,
        gt_num_atoms,
        gt_lengths=None,
        gt_angles=None,
        teacher_forcing=False,
    ):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing.
        """
        lengths_and_angles, lengths, angles = self.predict_lattice(
            z, gt_num_atoms
        )
        # Train stage
        if self.hparams.teacher_forcing_lattice and teacher_forcing:
            lengths = gt_lengths
            angles = gt_angles
        # Generate stage, i.e. langevin dynamics
        else:
            pass
        return (
            lengths_and_angles,
            lengths,
            angles,
        )

    @torch.no_grad()
    def langevin_dynamics(self, z, ld_kwargs, gt_num_atoms, gt_atom_types):
        """
        decode crystral structure from latent embeddings.
        ld_kwargs: args for doing annealed langevin dynamics sampling:
            n_step_each:  number of steps for each sigma level.
            step_lr:      step size param.
            min_sigma:    minimum sigma to use in annealed langevin dynamics.
            save_traj:    if <True>, save the entire LD trajectory.
            disable_bar:  disable the progress bar of langevin dynamics.
        gt_num_atoms: if not <None>, use the ground truth number of atoms.
        gt_atom_types: if not <None>, use the ground truth atom types.
        """
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        # obtain key stats.
        _, lengths, angles = self.decode_stats(z, gt_num_atoms)
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # obtain atom types.
        # composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        # if gt_atom_types is None:
        #     cur_atom_types = self.sample_composition(
        #         composition_per_atom, num_atoms
        #     )
        # else:
        #     cur_atom_types = gt_atom_types
        cur_atom_types = gt_atom_types

        # init coords.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)

        # annealed langevin dynamics.
        for sigma in tqdm(
            self.sigmas,
            total=self.sigmas.size(0),
            disable=ld_kwargs.disable_bar,
        ):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):
                noise_cart = torch.randn_like(cur_frac_coords) * torch.sqrt(
                    step_size * 2
                )
                pred_cart_coord_diff, pred_atom_types = self.decoder(
                    z,
                    cur_frac_coords,
                    cur_atom_types,
                    num_atoms,
                    lengths,
                    angles,
                )
                cur_cart_coords = frac_to_cart_coords(
                    cur_frac_coords, lengths, angles, num_atoms
                )
                pred_cart_coord_diff = pred_cart_coord_diff / sigma
                cur_cart_coords = (
                    cur_cart_coords
                    + step_size * pred_cart_coord_diff
                    + noise_cart
                )
                cur_frac_coords = cart_to_frac_coords(
                    cur_cart_coords, lengths, angles, num_atoms
                )

                if gt_atom_types is None:  # never used
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(
                        step_size * pred_cart_coord_diff
                    )
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        output_dict = {
            'num_atoms': num_atoms,
            'lengths': lengths,
            'angles': angles,
            'frac_coords': cur_frac_coords,
            'atom_types': cur_atom_types,
            'is_traj': False,
        }

        if ld_kwargs.save_traj:
            output_dict.update(
                dict(
                    all_frac_coords=torch.stack(all_frac_coords, dim=0),
                    all_atom_types=torch.stack(all_atom_types, dim=0),
                    all_pred_cart_coord_diff=torch.stack(
                        all_pred_cart_coord_diff, dim=0
                    ),
                    all_noise_cart=torch.stack(all_noise_cart, dim=0),
                    is_traj=True,
                )
            )

        return output_dict

    def sample(self, gt_num_atoms, gt_atom_types, num_samples, ld_kwargs):
        z = torch.randn(
            num_samples, self.hparams.hidden_dim, device=self.device
        )
        # cond z
        z = self.comp_cond(z, gt_atom_types, gt_num_atoms)
        samples = self.langevin_dynamics(
            z, ld_kwargs, gt_num_atoms, gt_atom_types
        )
        return samples

    def forward(self, batch, teacher_forcing=False, training=True):
        # hacky way to resolve the NaN issue. Will need more careful debugging later.
        mu, log_var, z = self.encode(batch)
        # z (B, lattent_dim)

        # conditional z
        cond_z = self.comp_cond(z, batch.atom_types, batch.num_atoms)

        # pred lattice from cond_z
        (
            pred_lengths_and_angles,  # (B, 6)
            pred_lengths,  # (B, 3)
            pred_angles,  # (B, 3)
        ) = self.decode_stats(
            cond_z,
            batch.num_atoms,
            batch.lengths,
            batch.angles,
            teacher_forcing,
        )

        # sample noise levels. noise on each atom
        noise_level = torch.randint(
            0,
            self.sigmas.size(0),
            (batch.num_atoms.size(0),),
            device=self.device,
        )
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            batch.num_atoms, dim=0
        )

        # add noise to the cart coords
        cart_noises_per_atom = (
            torch.randn_like(batch.frac_coords) * used_sigmas_per_atom[:, None]
        )
        cart_coords = frac_to_cart_coords(
            batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms
        )
        # N(X, σX^2 I) =    X    +    σX * N(0, 1)
        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(  # ~X
            cart_coords, pred_lengths, pred_angles, batch.num_atoms
        )

        pred_cart_coord_diff, _ = self.decoder(
            z,
            noisy_frac_coords,
            batch.atom_types,
            batch.num_atoms,
            pred_lengths,
            pred_angles,
        )

        # compute loss.
        # num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        # composition_loss = self.composition_loss(
        #     pred_composition_per_atom, batch.atom_types, batch
        # )
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch
        )
        # type_loss = self.type_loss(
        #     pred_atom_types, batch.atom_types, used_type_sigmas_per_atom, batch
        # )

        kld_loss = self.kld_loss(mu, log_var)

        if self.hparams.predict_property:
            property_loss = self.property_loss(z, batch)
        else:
            property_loss = 0.0

        return {
            # 'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            # 'composition_loss': composition_loss,
            'coord_loss': coord_loss,
            # 'type_loss': type_loss,
            'kld_loss': kld_loss,
            'property_loss': property_loss,
            # 'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            # 'pred_atom_types': pred_atom_types,
            # 'pred_composition_per_atom': pred_composition_per_atom,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            # 'rand_atom_types': rand_atom_types,
            'z': z,
        }

    def generate_rand_init(
        self,
        pred_composition_per_atom,
        pred_lengths,
        pred_angles,
        num_atoms,
        batch,
    ):
        rand_frac_coords = torch.rand(
            num_atoms.sum(), 3, device=num_atoms.device
        )
        pred_composition_per_atom = F.softmax(pred_composition_per_atom, dim=-1)
        rand_atom_types = self.sample_composition(
            pred_composition_per_atom, num_atoms
        )
        return rand_frac_coords, rand_atom_types

    def sample_composition(self, composition_prob, num_atoms):
        """
        Samples composition such that it exactly satisfies composition_prob
        """
        batch = torch.arange(
            len(num_atoms), device=num_atoms.device
        ).repeat_interleave(num_atoms)
        assert composition_prob.size(0) == num_atoms.sum() == batch.size(0)
        composition_prob = scatter(
            composition_prob, index=batch, dim=0, reduce='mean'
        )

        all_sampled_comp = []

        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            comp_num = torch.round(comp_prob * num_atom)
            atom_type = torch.nonzero(comp_num, as_tuple=True)[0] + 1
            atom_num = comp_num[atom_type - 1].long()

            sampled_comp = atom_type.repeat_interleave(atom_num, dim=0)

            # if the rounded composition gives less atoms, sample the rest
            if sampled_comp.size(0) < num_atom:
                left_atom_num = num_atom - sampled_comp.size(0)

                left_comp_prob = comp_prob - comp_num.float() / num_atom

                left_comp_prob[left_comp_prob < 0.0] = 0.0
                left_comp = torch.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True
                )
                # convert to atomic number
                left_comp = left_comp + 1
                sampled_comp = torch.cat([sampled_comp, left_comp], dim=0)

            sampled_comp = sampled_comp[torch.randperm(sampled_comp.size(0))]
            sampled_comp = sampled_comp[:num_atom]
            all_sampled_comp.append(sampled_comp)

        all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_property(self, z):
        self.scaler.match_device(z)
        return self.scaler.inverse_transform(self.fc_property(z))

    def predict_lattice(self, z, num_atoms):
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles
        )
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.hparams.data.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float() ** (
                1 / 3
            )
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z, num_atoms):
        # [z1, z2] repeat (2, 3) -> [z1, z1, z2, z2, z2]
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom

    def num_atom_loss(self, pred_num_atoms, batch):
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    def property_loss(self, z, batch):
        return F.mse_loss(self.fc_property(z), batch.y)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        if self.hparams.data.lattice_scale_method == 'scale_length':
            target_lengths = batch.lengths / batch.num_atoms.view(
                -1, 1
            ).float() ** (1 / 3)
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles], dim=-1
        )
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles
        )
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(
        self, pred_composition_per_atom, target_atom_types, batch
    ):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_composition_per_atom, target_atom_types, reduction='none'
        )
        return scatter(loss, batch.batch, reduce='mean').mean()

    def coord_loss(
        self,
        pred_cart_coord_diff,
        noisy_frac_coords,
        used_sigmas_per_atom,
        batch,
    ):
        noisy_cart_coords = frac_to_cart_coords(
            noisy_frac_coords, batch.lengths, batch.angles, batch.num_atoms
        )
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms
        )
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords,
            noisy_cart_coords,
            batch.lengths,
            batch.angles,
            batch.num_atoms,
            self.device,
            return_vector=True,
        )

        target_cart_coord_diff = (
            target_cart_coord_diff / used_sigmas_per_atom[:, None] ** 2
        )
        pred_cart_coord_diff = (
            pred_cart_coord_diff / used_sigmas_per_atom[:, None]
        )

        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff) ** 2, dim=1
        )

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def type_loss(
        self,
        pred_atom_types,
        target_atom_types,
        used_type_sigmas_per_atom,
        batch,
    ):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, reduction='none'
        )
        # rescale loss according to noise
        loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.batch, reduce='mean').mean()

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0,
        )
        return kld_loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch
        )
        outputs = self(batch, teacher_forcing, training=True)
        # print(torch.max(outputs['z']))
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
            batch_size=batch.num_graphs,
        )
        return loss

    def compute_stats(self, batch, outputs, prefix):
        # num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        # type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        # composition_loss = outputs['composition_loss']
        property_loss = outputs['property_loss']

        loss = (
            # self.hparams.cost_natom * num_atom_loss
            +self.hparams.cost_lattice * lattice_loss
            + self.hparams.cost_coord * coord_loss
            # + self.hparams.cost_type * type_loss
            + self.hparams.beta * kld_loss
            # + self.hparams.cost_composition * composition_loss
            + self.hparams.cost_property * property_loss
        )
        assert torch.isfinite(loss)

        log_dict = {
            f'{prefix}_loss': loss,
            # f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            # f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
            # f'{prefix}_composition_loss': composition_loss,
        }

        if prefix != 'train':
            # validation/test loss only has coord and type
            loss = (
                self.hparams.cost_coord
                * coord_loss
                # + self.hparams.cost_type * type_loss
            )

            # evaluate num_atom prediction.
            # pred_num_atoms = outputs['pred_num_atoms'].argmax(dim=-1)
            # num_atom_accuracy = (
            #     pred_num_atoms == batch.num_atoms
            # ).sum() / batch.num_graphs

            # evalute lattice prediction.
            pred_lengths_and_angles = outputs['pred_lengths_and_angles']
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles
            )
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * batch.num_atoms.view(
                    -1, 1
                ).float() ** (1 / 3)
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)

            # evaluate atom type prediction.
            # pred_atom_types = outputs['pred_atom_types']
            # target_atom_types = outputs['target_atom_types']
            # type_accuracy = pred_atom_types.argmax(dim=-1) == (
            #     target_atom_types - 1
            # )
            # type_accuracy = scatter(
            #     type_accuracy.float(), batch.batch, dim=0, reduce='mean'
            # ).mean()

            log_dict.update(
                {
                    f'{prefix}_loss': loss,
                    f'{prefix}_property_loss': property_loss,
                    # f'{prefix}_natom_accuracy': num_atom_accuracy,
                    f'{prefix}_lengths_mard': lengths_mard,
                    f'{prefix}_angles_mae': angles_mae,
                    f'{prefix}_volumes_mard': volumes_mard,
                    # f'{prefix}_type_accuracy': type_accuracy,
                }
            )

        return log_dict, loss


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('fit')
    batch = next(iter(datamodule.train_dataloader()))
    print(batch)
    # =========================================
    comp_cond = CompositionCondition(cfg.model.hidden_dim, cfg.model.latent_dim)
    z = torch.zeros((batch.num_graphs, cfg.model.latent_dim))
    print(comp_cond(z, batch.atom_types, batch.num_atoms).shape)
    cdvae = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    cdvae.lattice_scaler = datamodule.lattice_scaler.copy()
    cdvae.scaler = datamodule.scaler.copy()
    # -----------------
    # trainer = pl.Trainer(fast_dev_run=True)
    # trainer.fit(model=cdvae, datamodule=datamodule)
    cdvae.to('cuda')
    for val_loader in datamodule.val_dataloader():
        for idx, batch in enumerate(val_loader):
            batch.to('cuda')
            returns = cdvae(batch)
            z = returns['z']
            print(idx, z.shape, z.max().item())
    # Debug sample
    from pymatgen.core.composition import Composition
    from itertools import chain

    comp = Composition('H2O')
    n_sample = 10
    each_atom_types = list(
        chain.from_iterable(
            [elem.number] * int(n)
            for elem, n in Composition(
                Composition('H2O').get_integer_formula_and_factor()[0]
            ).items()
        )
    )
    num_atoms = torch.tensor(
        [len(each_atom_types)] * n_sample, device=cdvae.device
    )
    atom_types = torch.tensor(each_atom_types * n_sample, device=cdvae.device)
    from types import SimpleNamespace

    ld_kwargs = SimpleNamespace(
        n_step_each=10,
        step_lr=1e-4,
        min_sigma=0,
        save_traj=False,
        disable_bar=False,
    )
    gen_samples = cdvae.sample(num_atoms, atom_types, n_sample, ld_kwargs)
    print(gen_samples['frac_coords'].shape)


if __name__ == '__main__':
    main()
