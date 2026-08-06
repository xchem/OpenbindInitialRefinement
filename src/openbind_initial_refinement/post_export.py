import argparse
from pathlib import Path
from typing import TypedDict
import logging
import pathlib

import numpy as np
import gemmi

from openbind_initial_refinement.initial_refinement import drop_atoms, ResID

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Constants:
    COMMON_F_PHI_LABEL_PAIRS = (
    ("FWT", "PHWT"),
    ("2FOFCWT", "PH2FOFCWT"),
    ("2FOFCWT_iso-fill", "PH2FOFCWT_iso-fill"),
    ("2FOFCWT_fill", "PH2FOFCWT_fill",),
    ("2FOFCWT", "PHI2FOFCWT")
)

C = Constants

class StructureFactors(TypedDict):
    F: str
    PHI: str
    
class Dataset(TypedDict):
    structure: gemmi.Structure
    reflections: gemmi.Mtz
    sfs: StructureFactors

class ResID(TypedDict):
    chain: str
    insertion: str
    conf: str

def read_structure(path) -> gemmi.Structure:
    return gemmi.read_structure(str(path))

def read_mtz(path) -> gemmi.Mtz:
    return gemmi.read_mtz_file(str(path))

def read_dataset_from_dir(path) -> Dataset:
    st = read_structure(path / 'refine.pdb')
    mtz = read_mtz(path / 'refine.mtz')
    labels = mtz.column_labels()
    fs = {f: j for j, (f, phi) in enumerate(C.COMMON_F_PHI_LABEL_PAIRS)}
    sfs = None
    for f, phi in C.COMMON_F_PHI_LABEL_PAIRS:
        if (f in labels) & (phi in labels):
            sfs = {'F': f, 'PHI': phi}
    if not sfs:
        raise Exception(f'No obvious F column in refine.mtz at {path}')
    return {
        'structure': st,
        'reflections': mtz,
        'sfs': sfs
    }

def remove_weak_waters(dataset: Dataset, output_path, threshold: float = 1.0) -> gemmi.Structure:
    # Open the map
    grid = dataset['reflections'].transform_f_phi_to_map(dataset['sfs']['F'], dataset['sfs']['PHI'], sample_rate=4)

    std = np.std(np.array(grid))
    logger.info(f'Map std is: {round(std, 2)}')

    # Sample map at each water position
    scores = {}
    for model in dataset['structure']:
        for chain in model:
            for res in chain:
                if res.name != 'HOH':
                    continue
                for atom in res:
                    scores[ResID(chain.name, str(res.seqid.num))] = grid.interpolate_value(atom.pos) / std

    # Drop waters below threshold
    to_drop = [x for x in scores if scores[x] < threshold]
    logger.info(f'Waters to drop: {to_drop}')
    st = drop_atoms(dataset['structure'], to_drop)

    st.write_minimal_pdb(str(output_path))
    return st


def reciprocal_space_refinement():
    ...


def get_density_fit_stats():
    ...


def get_structural_model_stats():
    ...


def main(path):
    # 
    path = path.Pathlib(path)
    dataset = read_dataset_from_dir(path)
    logger.info(f'Got dataset in {path}')

    # Get rid of weak waters with findwaters
    desolv_path = path / f'refine_desolv.pdb'
    remove_weak_waters(dataset, desolv_path)
    logger.info(f'Wrote st w/o weak water to {desolv_path}')

    # Do an initial reciprocal refinement
    refined_pdb_path = path / 'autorefine.pdb'
    refined_mtz_path = path / 'autorefine.mtz'
    new_structure_path = reciprocal_space_refinement(desolv_path, refined_pdb_path, refined_mtz_path)
    logger.info(f'Refined pdb/mtz to {refined_pdb_path}/{refined_mtz_path}')

    # Check for bad density with gemmi.blobs
    density_fit_stats = get_density_fit_stats(new_structure_path)

    # Check for bad model with molprobity 
    structural_model_stats = get_structural_model_stats(new_structure_path)

    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_building_dataset_dir'
    )
    args = parser.parse_args()
    main(pathlib.Path(args.model_building_dataset_dir))
