import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write


def get_f_set(atoms: Atoms) -> set:
    formula = atoms.symbols.formula
    f_set = set(formula.count().keys())
    return f_set

def valid_hcno(atoms: Atoms) -> bool:
    f_set = get_f_set(atoms)
    return f_set == set(['H', 'C', 'N', 'O'])

def get_min_dist(atoms: Atoms) -> float:
    all_dist: np.ndarray = atoms.get_all_distances(mic=True)
    min_dist = all_dist[all_dist.nonzero()].min()
    return min_dist

if __name__ == '__main__':
    d = sys.argv[1]  # directory containing structure files
    v_path = Path('valid_hcno')
    v_path.mkdir(exist_ok=True)
    ser_list = []
    for f in Path(d).glob('*.vasp'):
        atoms = read(f)
        if valid_hcno(atoms):
            min_dist = get_min_dist(atoms)
            ser = pd.Series({'name': f.name, 'min_dist': min_dist})
            ser_list.append(ser)
            write(v_path.joinpath(f.name), atoms, direct=True)
    print(len(ser_list))
    df = pd.DataFrame(ser_list)
    df = df.sort_values(by='min_dist', ascending=False)
    df.to_csv('valid_hcno.txt', index=False, sep='\t')
