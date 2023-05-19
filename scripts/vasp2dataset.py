import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
from ase.io import read, write
from tqdm import tqdm


def atoms2cifstring(atoms):
    with io.BytesIO() as buffer, redirect_stdout(buffer):
        write('-', atoms, format='cif')
        cif = buffer.getvalue().decode()  # byte to string
    return cif


def read_outcar(outcar):
    try:
        atoms = read(outcar, format="vasp-out")
    except Exception:
        print(outcar, "NOT complete")
        return False
    energy = atoms.get_potential_energy()  # energy sigma -> 0
    with open(outcar, "r") as f:
        for line in f.readlines()[::-1]:
            if "P V" in line:
                pv = float(line.strip().split()[-1])  # eV
    enthalpy = energy + pv
    # _id = Path(outcar).stem[7:]  # OUTCAR_*.vasp
    _id = Path(outcar).parent.stem + '.vasp'  # *.run/OUTCAR_ -> *.vasp
    return {
            "material_id": _id,
            "formula": atoms.get_chemical_formula("metal"),
            "nsites": len(atoms),
            "volume": atoms.get_volume(),
            "energy": energy,
            "enthalpy": enthalpy,
            "energy_per_atom": energy / len(atoms),
            "enthalpy_per_atom": enthalpy / len(atoms),
            "cif": atoms2cifstring(atoms),
        }


def main(dir_list):
    ser_list = []
    for d in dir_list:
        for f in tqdm(Path(d).glob("*.run/OUTCAR")):
            data_dict = read_outcar(f)
            if data_dict:
                ser = pd.Series(data_dict)
            ser_list.append(ser)
    df = pd.DataFrame(ser_list)
    return df


if __name__ == "__main__":
    dir_list = sys.argv[1:]
    df = main(dir_list)
    df.to_feather("opt.feather")
