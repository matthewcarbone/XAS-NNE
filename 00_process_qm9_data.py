from datetime import datetime
from pathlib import Path
import pickle
import time

from pymatgen.core.structure import Molecule
from tqdm import tqdm

from xas_nne.utils import read_json, save_json, timeit
from xas_nne.qm9 import read_qm9_xyz
from xas_nne.feff import FeffWriter, load_completed_FEFF_results


DSGDB9NSD_PATH = None
DATA_ROOT = Path("data") / Path("qm9")
QM9_TARGET_JSON = DATA_ROOT / Path("raw_qm9.json")


STEP_QM9_TO_JSON = True
STEP_FEFF_INPUTS = False
ABSORBERS = ["C", "N", "O"]
SPECTRUM_TYPE = "XANES"
STEP_PICKLE = True


assert SPECTRUM_TYPE in ["XANES", "EXAFS"]

if __name__ == '__main__':

    if STEP_QM9_TO_JSON:
        with timeit("qm9 -> json"):
            print("QM9 -> parsed json exists: ", QM9_TARGET_JSON.exists())
            if QM9_TARGET_JSON.exists():
                data = read_json(QM9_TARGET_JSON)
            else:
                print("QM9 directory exists:       ", DSGDB9NSD_PATH.exists())
                if not DSGDB9NSD_PATH.exists():
                    raise ValueError("Incorrect path for QM9 database")
                data = dict()
                paths = list(DSGDB9NSD_PATH.iterdir())
                t0 = time.time()
                for file in tqdm(paths):
                    if "dsgdb9nsd" not in str(file):
                        continue
                    d = read_qm9_xyz(file)
                    molecule = Molecule.from_file(file)
                    qm9id = d.pop("qm9id")
                    d.pop("xyz")
                    d.pop("elements")
                    d["molecule"] = molecule.as_dict()
                    data[qm9id] = d
                save_json(data, QM9_TARGET_JSON, sort_keys=False)

    if STEP_FEFF_INPUTS:
        with timeit("FEFF inputs"):
            # Initialize a Molecule object from the serialized version
            molecules = {
                key: Molecule.from_dict(
                    xx["molecule"]
                ) for key, xx in data.items()
            }
            xanes = True if SPECTRUM_TYPE == "XANES" else False
            for qm9id, molecule in tqdm(molecules.items()):
                for absorber in ABSORBERS:
                    f = FeffWriter(molecule, xanes=xanes, name=qm9id)
                    if absorber not in f.elements:
                        continue
                    dname = Path(f"{absorber}-{SPECTRUM_TYPE}") \
                        / Path(f"{int(qm9id):06}")
                    f.write_feff_inputs(str(dname), absorber=absorber)

    # This step can take quite a while...
    if STEP_PICKLE:
        today = datetime.now().strftime("%y%m%d")
        absorbers_string = "-".join(ABSORBERS)
        save_string = DATA_ROOT / Path(f"XANES-{today}-{absorbers_string}.pkl")
        if save_string.exists():
            raise ValueError(f"Pickle file {save_string} already exists")

        print(f"To-pickle running, saving to {save_string}")
        for absorber in ABSORBERS:
            feff_inp_files = (DATA_ROOT / Path(
                f"{absorber}-{SPECTRUM_TYPE}"
            )).rglob("feff.inp")
            failures = []
            total = 0
            t0 = time.time()
            for ii, path in enumerate(feff_inp_files):
                data_dict = load_completed_FEFF_results(path.parent)
                qm9_id = data_dict.pop("qm9id")
                site = data_dict.pop("site")
                absorbing_atom = data_dict.pop("absorbing_atom")
                key = f"{site}_{absorbing_atom}"

                if "xanes" not in data[qm9_id].keys():
                    data[qm9_id]["xanes"] = dict()

                data[qm9_id]["xanes"][key] = data_dict

                if data_dict["spectrum"] is None:
                    failures.append(path.parent)
                    continue

                total += 1

                if ii % 10000 == 0:
                    dt = (time.time() - t0) / 60.0
                    print(f"{ii:08} done {dt:.1f} m")

            # Sometimes FEFF fails to converge. We will ignore these cases
            # later.
            print(
                f"For absorber {absorber} we note {len(failures)} failures "
                f"out of {total} total"
            )

        pickle.dump(
            data, open(save_string, "wb"), protocol=pickle.HIGHEST_PROTOCOL
        )
