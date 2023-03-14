import pickle
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
from omegaconf import ValueNode
from torch.utils.data import Dataset
from torch_geometric.data import Data

from cdvae.common.data_utils import (
    add_scaled_lattice_prop,
    preprocess,
    preprocess_tensors,
)
from cdvae.common.utils import PROJECT_ROOT


class CrystDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,  # original crystal info
        save_path: ValueNode,  # processed graph data
        force_process: ValueNode,  # process or load
        prop: ValueNode,  # list
        niggli: ValueNode,
        primitive: ValueNode,
        graph_method: ValueNode,
        preprocess_workers: ValueNode,
        lattice_scale_method: ValueNode,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.save_path = save_path
        self.force_process = force_process
        self.name = name
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        if self.force_process or not Path(self.save_path).exists():
            self.cached_data = preprocess(
                self.path,
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                prop_list=prop,
            )
            hydra.utils.log.info(f"Dump into {self.save_path} ...")
            pickle.dump(self.cached_data, open(self.save_path, 'wb'))
        else:
            hydra.utils.log.info(f"Load from {self.save_path} ...")
            self.cached_data = pickle.load(open(self.save_path, 'rb'))

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.prop_scalers: list = None  # list of prop_scaler

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index) -> Data:
        data_dict = self.cached_data[index]
        # {'mp_id', 'cif', 'graph_array' **prop}

        # scaler is set in DataModule set stage
        p_dict = {
            p: prop_scaler.transform(torch.tensor(data_dict[p]).view(1, -1))
            for p, prop_scaler in zip(self.prop, self.prop_scalers, strict=True)
        }

        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,
            to_jimages,
            num_atoms,
        ) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(edge_indices.T).contiguous(),
            # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=
            num_atoms,  # special attribute used for batching in pytorch geometric
            mp_id=data_dict['mp_id'],
            **p_dict,
        )
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=}, {self.save_path=})"


class TensorCrystDataset(Dataset):
    def __init__(
        self,
        crystal_array_list,
        niggli,
        primitive,
        graph_method,
        preprocess_workers,
        lattice_scale_method,
        **kwargs,
    ):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
        )

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,
            to_jimages,
            num_atoms,
        ) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=
            num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"
