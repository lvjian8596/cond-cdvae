import warnings
from collections import defaultdict
from itertools import chain
from pathlib import Path

import click
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from statgen import to_format_table
from tqdm import tqdm


def rglobvasp(fdir: Path):
    fdir = Path(fdir)
    return list(
        chain(
            fdir.rglob("POSCAR"),
            fdir.rglob("POSCAR_*"),
            fdir.rglob("*.vasp"),
            fdir.rglob("poscar_*"),
            fdir.rglob("contcar_*"),
        )
    )


class HydrideAnalyzer:
    def __init__(self, structure: Structure, max_Hbond=1.0):
        self.CrystalNN = local_env.CrystalNN(
            x_diff_weight=-1,
            porous_adjustment=False,
        )
        self.structure = structure
        self.natoms = self.nsites = len(structure)
        self.mindist = np.min(self.distmat)
        self.nH = len(self.Hindices)
        self.nbondedH = self.get_nbondedH(structure, self.distmat, max_Hbond)
        self.nMcageH = len(self.McageH)

    @property
    def series(self):
        info = {}
        info.update(self.spgdict)
        keys_direct = ["formula_hill", "nsites", "mindist", "nH", "nbondedH", "nMcageH"]
        info.update({key: getattr(self, key) for key in keys_direct})
        info.update(
            {
                f"min_H-M{i}": np.min(self.dist_HM[k])
                for i, k in enumerate(sorted(self.dist_HM.keys()))
            }
        )
        info.update(
            {
                f"avg_H-M{i}": np.mean(self.dist_HM[k])
                for i, k in enumerate(sorted(self.dist_HM.keys()))
            }
        )
        info.update(
            {
                f"std_H-M{i}": np.std(self.dist_HM[k])
                for i, k in enumerate(sorted(self.dist_HM.keys()))
            }
        )
        return pd.Series(info)

    @property
    def spgdict(self):
        symkeys = ["5e-01", "1e-01", "1e-02"]
        symprecs = [0.5, 0.1, 0.01]
        if getattr(self, "_spgdict", None) is None:
            self._spgdict = {
                k: SpacegroupAnalyzer(self.structure, prec).get_space_group_number()
                for k, prec in zip(symkeys, symprecs)
            }
        return self._spgdict

    @property
    def Hindices(self):
        """indices of H"""
        if getattr(self, "_Hindices", None) is None:
            self._Hindices = [
                i for i, n in enumerate(self.structure.atomic_numbers) if n == 1
            ]
        return self._Hindices

    @property
    def Mindices(self):
        """indices of non-H"""
        if getattr(self, "_Mindices", None) is None:
            self._Mindices = [
                i for i, n in enumerate(self.structure.atomic_numbers) if n != 1
            ]
        return self._Mindices

    def get_nbondedH(self, structure: Structure, distmat, max_Hbond=1.0):
        Hdistmat = np.take(distmat, self._Hindices, 0)
        return np.any(Hdistmat <= max_Hbond, axis=1).sum()

    @property
    def distmat(self):
        if getattr(self, "_distmat", None) is None:
            self._distmat = self.structure.distance_matrix
            np.fill_diagonal(self._distmat, min(self.structure.lattice.abc))
        return self._distmat

    @property
    def formula_hill(self):
        if getattr(self, "_formula_hill", None) is None:
            atoms = AseAtomsAdaptor.get_atoms(self.structure)
            self._formula_hill = atoms.get_chemical_formula(mode="hill")
        return self._formula_hill

    @property
    def graph(self):
        """[[i, j, jix, jiy, jiz], ...]"""
        if getattr(self, "_graph", None) is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                graph = StructureGraph.with_local_env_strategy(
                    self.structure, self.CrystalNN
                ).graph
            edges = np.array(
                [
                    [i, j] + list(to_jimage)
                    for i, j, to_jimage in graph.edges(data='to_jimage')
                ],
                dtype=int,
            )
            # reverse edge direction and flip to_jimage
            rev_edges = edges[:, [1, 0, 2, 3, 4]] * np.array([[1, 1, -1, -1, -1]])
            self._graph = np.concatenate([edges, rev_edges])
        return self._graph

    @property
    def Mcages(self):
        """dict of each M's cage vertexes, {Mi: [[j, ...], ...]}"""
        if getattr(self, "_Mcages", None) is None:
            self._Mcages = {
                Mi: [j for i, *j in self.graph if i == Mi] for Mi in self.Mindices
            }
        return self._Mcages

    @property
    def McageH(self):
        "H indices surrounding M"
        if getattr(self, "_McageH", None) is None:
            cage_vertexes = [
                vertex[0] for Mcage in self.Mcages.values() for vertex in Mcage
            ]
            self._McageH = list(set(self.Hindices) & set(cage_vertexes))
        return self._McageH

    @property
    def dist_HM(self):
        if getattr(self, "_d_HM", None) is None:
            self._d_HM = defaultdict(list)
            for Mi, Mcage in self.Mcages.items():
                Hilist = [j for j, *_ in Mcage if j in self.Hindices]
                d_MH_list = [self.distmat[Mi, Hi] for Hi in Hilist]
                self._d_HM[f"d_H-{self.structure[Mi].species_string}"] += d_MH_list
        return self._d_HM


def wrapped_analysis(fvasp, max_Hbond):
    structure = Poscar.from_file(fvasp).structure
    analyzer = HydrideAnalyzer(structure, max_Hbond)
    return analyzer.series


def analysis(njobs, vaspdirlist: list[Path], max_Hbond):
    for vaspdir in vaspdirlist:
        vaspflist = rglobvasp(vaspdir)
        # structures = [Poscar.from_file(fvasp).structure for fvasp in vaspflist]
        series_list = Parallel(njobs, "multiprocessing")(
            delayed(wrapped_analysis)(fvasp, max_Hbond)
            for fvasp in tqdm(
                vaspflist, desc=str(vaspdir)[-20:], ncols=120, mininterval=1
            )
        )
        namelist = [str(fvasp.relative_to(vaspdir.parent)) for fvasp in vaspflist]
        df = pd.DataFrame(series_list)
        df.insert(0, "name", namelist)
        with open(vaspdir.parent.joinpath(f"{vaspdir.name}.hydride.table"), "w") as f:
            f.write(to_format_table(df))
        print(df)


@click.command()
@click.argument('vaspdir', nargs=-1)
@click.option("-j", "--njobs", type=int, default=1)
@click.option("--max_Hbond", type=float, default=1.0, help="max H bond (default 1.0)")
def main(vaspdir, njobs, max_hbond):
    vaspdir = [Path(d) for d in vaspdir]
    analysis(njobs, vaspdir, max_hbond)


if __name__ == "__main__":
    main()
