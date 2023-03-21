import pickle
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

# a_list = pickle.load(open(gen_pt_file.with_name('gen.pkl'), 'rb'))


def get_distances(atoms):
    d_m = atoms.get_all_distances(mic=True)
    d_m = np.triu(d_m)
    return d_m[d_m > 0]


def valid_Si38(atoms):
    if len(atoms) != 38:
        return False
    if any(map(lambda atom: atom.number != 14, atoms)):
        return False

    return True


if __name__ == '__main__':
    d = sys.argv[1]  # directory containing structure files
    v = Path(d).with_name('valid_gen')
    v.mkdir(exist_ok=True)

    for f in Path(d).glob('*.vasp'):
        atoms = read(f)
        if valid_Si38(atoms):
            write(v.joinpath(f.name), atoms)
