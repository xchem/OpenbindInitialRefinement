import argparse
import dataclasses
import pathlib
import subprocess
import re
import yaml
import shutil
import logging

import numpy as np
import gemmi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclasses.dataclass
class ResID:
    chain: str
    seqid: str

    def __repr__(self):
        return f'{self.chain}/{self.seqid}'

@dataclasses.dataclass
class ResidueSubsequence:
    chain: str
    min: str
    max: str

    def __repr__(self):
        return f'{self.chain}:{self.min}-{self.max}'

class Constants:
    real_space_refine_script = "module load ccp4; coot-mini-rsr --pdbin {pdbin} --mapin {mapin} --dictin {dictin} --pdbout {pdbout} --chain-id {chainid} --resno-start {resnostart} --resno-end {resnoend}"
    process_dataset_file = 'processed_dataset.yaml'
    events_file = 'events.yaml'
    event_map_file_pattern = "{dtag}-event_{event_idx}_1-BDC_{bdc}_map.native.ccp4"
    modelled_st_dir = 'modelled_structures'
    pandda_model_path = '{dtag}-pandda-model.pdb'
    mean_map_path = "{dtag}-ground-state-average-map.native.ccp4"
    xmap_path='xmap.ccp4'
    backup_st_path = 'backup.pdb'
    recalculated_event_map_file = 'event_map_recalculated.ccp4'
    key_summary = 'Summary'
    key_res = 'Processing Resolution'
    key_models = 'Models'
    key_bdc = 'BDC'
    key_selected_model = 'Selected Model'
    key_score = 'score'
    key_centroid = 'Centroid'
    key_build = 'Build'
    key_ligand_id = 'Ligand Key'
    compound_dir = 'ligand_files'
    resnames_to_ignore = [
        'LIG',
        'XXX',
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "GLY",
    ]
    intermediary_pdb_file = 'intermediary.pdb'
    output_pdb_file = 'rsr.pdb'

C = Constants


def get_ligand_resids(pandda_model_path):
    st = gemmi.read_structure(str(pandda_model_path))
    ligand_resids = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.name in ['LIG', "XXX"]:
                    ligand_resids.append(
                        ResID(chain.name, str(res.seqid.num))
                    )
    return ligand_resids


def backup_pandda_model(pandda_model_path, backup_path):
    if backup_path.exists():
        raise Exception(f'Already have a backup at {backup_path}: not safe to proceed as may delete data!')
    shutil.copyfile(pandda_model_path, backup_path)
    return backup_path

def get_ligand_centroid(st_path, ligand_resid):
    st = gemmi.read_structure(str(st_path))
    for model in st:
        for chain in model:
            if chain.name != ligand_resid.chain:
                continue
            for res in chain:
                if str(res.seqid.num) != ligand_resid.seqid:
                    continue
                poss = []
                for atom in res:
                    pos = atom.pos
                    poss.append([pos.x, pos.y, pos.z])
                centroid = np.mean(
                    poss,
                    axis=0
                    )
                return centroid

    raise Exception(f'No ligand found at {ligand_resid.chain}/{ligand_resid.seqid} in {st_path}')

def get_dist(ligand_centroid, centroid):
    return np.linalg.norm(np.array(ligand_centroid) - np.array(centroid))


def get_closest_event(st_path, ligand_resid, pandda_dataset_dir):
    with open(pandda_dataset_dir / C.events_file, 'r') as f:
        meta = yaml.safe_load(f) 

    # Get ligand centroid
    ligand_centroid = get_ligand_centroid(st_path, ligand_resid)
    logger.info(f'Got ligand centroid: {ligand_centroid}')

    # Get closest event
    dists = {}
    for event_id, event in meta.items():
        centroid = event[C.key_centroid]

        dists[event_id] = round(get_dist(ligand_centroid, centroid), 2)

    logger.info(f'Got distances to ligand: {dists}')

    # Get closest event (and hence event map) to ligand
    clostest_event = min(dists, key=lambda _x: dists[_x])
    closest_event_map_path = pandda_dataset_dir / C.event_map_file_pattern.format(
        dtag=pandda_dataset_dir.parts[-1], 
        event_idx=clostest_event, 
        bdc=round(1-meta[clostest_event][C.key_bdc]),
        )
    clostest_event_build_key = meta[clostest_event][C.key_build][C.key_ligand_id]

    return clostest_event, closest_event_map_path, clostest_event_build_key


def recalculate_event_map(event_map_path, xmap_path, mean_map_path):
    name = event_map_path.name
    bdc = float(re.search(
        r'1-BDC_([^_]+)',
        name,
    )[1])
 
    mean_map_ccp4 = gemmi.read_ccp4_map(str(mean_map_path), )
    mean_map_ccp4.setup(0.0)
    xmap_ccp4 = gemmi.read_ccp4_map(str(xmap_path), )
    xmap_ccp4.setup(0.0)

    mean_map_grid = mean_map_ccp4.grid
    xmap_grid = xmap_ccp4.grid

    mean_map_array = np.array(mean_map_grid, copy=False)
    xmap_array = np.array(xmap_grid, copy=False)

    event_map_array = (xmap_array - ((1-bdc) * mean_map_array)) / (bdc)

    event_map_grid = gemmi.FloatGrid(*[xmap_grid.nu, xmap_grid.nv, xmap_grid.nw])
    event_map_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    event_map_grid.set_unit_cell(xmap_grid.unit_cell)
    event_map_grid_array = np.array(event_map_grid, copy=False)
    event_map_grid_array[:, :, :] = event_map_array[:, :, :]

    return event_map_grid


def write_map(grid, path):
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(path))




def drop_atoms(st, resids: list[ResID]) -> gemmi.Structure:
    new_st = st.clone()
    for chain in st[0]:
        del new_st[0][chain.name]

    resids_hashable = [(resid.chain, resid.seqid) for resid in resids]
    
    for chain in st[0]:
        new_chain = gemmi.Chain(chain.name)
        for res in chain:
            if (chain.name, str(res.seqid.num)) in resids_hashable:
                    continue
            else:
                new_chain.add_residue(res)
        new_st[0].add_chain(new_chain)
    
    return new_st

def remove_ground_state_ligands(pdbin, output_path):
    st = gemmi.read_structure(str(pdbin))

    # Get non-lig, non-protein resids  
    atoms_to_drop = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.name not in C.resnames_to_ignore:
                    atoms_to_drop.append(
                        ResID(chain.name, str(res.seqid.num))
                    )

    logger.info(f'Dropping residues: {atoms_to_drop}')

    # Drop atoms
    new_st = drop_atoms(st, atoms_to_drop)

    # Write
    new_st.write_minimal_pdb(str(output_path))

    return output_path


def get_residue_sequences_around_ligand(st_path, resid, radius=10.0):
    # Break up the residues in a radius around ligand into continuous subsequences
    st = gemmi.read_structure(str(st_path))

    # Neighbour search
    ns = gemmi.NeighborSearch(st[0], st.cell, radius).populate()

    # For ligand atom get nearby atom resids
    resids = {}
    for model in st:
        for chain in model:
            if chain.name != resid.chain:
                continue
            for res in chain:
                if str(res.seqid.num) != resid.seqid:
                    continue
                for atom in res:
                    neighbours = ns.find_neighbors(atom, min_dist=0.1, max_dist=radius)
                    for mark in neighbours:
                        cra = mark.to_cra(st[0])
                        chain_name, seqid = cra.chain.name, str(cra.residue.seqid.num)
                        if seqid == res.seqid:
                            continue
                        resids[(chain_name, seqid)] = ResID(chain_name, seqid)
    logger.info(f'Resids near ligand: {[x for x in resids.values()]}')


    # mask each chain, extract continuous runs
    subsequences = []
    chains = set([resid.chain for resid in resids.values()])
    for chain in chains:
        chain_resids = sorted([int(resid.seqid) for resid in resids.values() if resid.chain == chain])
        print(chain_resids)
        diffs = np.ediff1d(chain_resids)
        print(diffs)
        discontinuity_indicies = np.where(diffs != 1)
        print(discontinuity_indicies)
        chain_subsequences = np.split(chain_resids, discontinuity_indicies[0]+1)
        for chain_subsequence in chain_subsequences:
            subsequences.append(
                ResidueSubsequence(
                    chain,
                    chain_subsequence[0],
                    chain_subsequence[-1]
                )
            )

    return subsequences

    ...

def real_space_refine(
        pdbin,
        mapin,
        dictin,
        pdbout,
        chain, 
        subsequence_min, 
        subsequence_max):
    script = C.real_space_refine_script.format(
            pdbin=pdbin,
            mapin=mapin,
            dictin=dictin,
            pdbout=pdbout,
            chainid=chain,
            resnostart=subsequence_min,
            resnoend=subsequence_max
    )    
    logger.info(f'Refinement script is: {script}')
    p = subprocess.Popen(
        script,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = p.communicate()
    logger.info(f'Stdout: {stdout}')
    logger.info(f'Stderr: {stderr}')


def run_real_space_refine(
        pdbin,
        mapin,
        dictin,
        pdbout,
        resid,
        intermediary_st_path
        ):

    # Strip ground state ligands
    logger.info(f'For ligand {resid} ligandless intermediary at {intermediary_st_path}', )
    remove_ground_state_ligands(pdbin, intermediary_st_path)

    # Get continuous subsequences
    subsequences = get_residue_sequences_around_ligand(intermediary_st_path, resid)
    logger.info(f'Got subsequences to refine: {subsequences}')

    # Coot refine
    for subsequence in subsequences:
        if pdbout.exists():
            pdbin = pdbout
        real_space_refine(
            pdbin,
            mapin,
            dictin,
            pdbout,
            subsequence.chain, 
            subsequence.min, 
            subsequence.max,
            )

    return pdbout


def main(pandda_dataset_dir):
    logger.info(f'PanDDA Dataset Dir: {pandda_dataset_dir}')

    # Get some important info
    dtag = pandda_dataset_dir.parts[-1]
    pandda_model_path = pandda_dataset_dir / C.modelled_st_dir / C.pandda_model_path.format(dtag=dtag)
    xmap_path = pandda_dataset_dir / C.xmap_path
    mean_map_path = pandda_dataset_dir / C.mean_map_path.format(dtag=dtag)

    logger.info(f'Dtag: {dtag}')
    logger.info(f'PanDDA Model Path: {pandda_model_path}')
    logger.info(f'Xmap Path: {xmap_path}')
    logger.info(f'Mean map path: {mean_map_path}')


    # Get ligand resids
    ligand_resids = get_ligand_resids(pandda_model_path)
    logger.info(f'Ligand Resids: {ligand_resids}')


    # Refine around each ligand
    for j, ligand_resid in enumerate(ligand_resids):
        logger.info(f'Processing ligandid: {ligand_resid}')

        # Make a model backup
        # backup_path = pandda_dataset_dir / C.modelled_st_dir / C.backup_st_path
        # logger.info(f'Backing up {pandda_model_path} to {backup_path}')
        # backup_pandda_model(pandda_model_path, backup_path)

        # Get ligand key and event id
        closest_event, event_map_path, ligand_key = get_closest_event(
            pandda_model_path, 
            ligand_resid, 
            pandda_dataset_dir,
            )
        logger.info(f'Closest event is: {closest_event}')
        logger.info(f'Closest event map is: {event_map_path}')
        logger.info(f'Ligand key is {ligand_key}')

        # Get ligand cif path
        ligand_cif_path = pandda_dataset_dir / C.compound_dir / f'{ligand_key}.cif'
        logger.info(f'Ligand cif path is: {ligand_cif_path}')

        # Uncut the event map 
        recalculated_event_map_path = pandda_dataset_dir / C.recalculated_event_map_file
        recalculated_event_map = recalculate_event_map(event_map_path, xmap_path, mean_map_path)
        write_map(recalculated_event_map, recalculated_event_map_path)
        logger.info(f'Recaclulating event map to: {recalculated_event_map_path}')

        # Real space refine against bound state with coot-mini-rsr
        output_pdb_path = pandda_dataset_dir / C.modelled_st_dir / C.output_pdb_file
        if output_pdb_path.exists():
            pdbin = output_pdb_path
        else:
            pdbin = pandda_model_path
        intermediary_st_path = pandda_dataset_dir / C.modelled_st_dir / C.intermediary_pdb_file
        logger.info(f'Refining {pdbin} to {output_pdb_path}')
        run_real_space_refine(
            pdbin=pdbin,
            mapin=recalculated_event_map_path,
            dictin=ligand_cif_path,
            pdbout=output_pdb_path,
            resid=ligand_resid,
            intermediary_st_path=intermediary_st_path
        )

        # Copy new structure to pandda_model_path
        # logger.info(f'Copying ')
        # shutil.copyfile(new_structure_path, )

        # TODO: Delete recalculated event map - is large

        # Get rid of residues that didn't move
        # new_structure_path = reset_unmoved_residues()
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'pandda_dataset_dir'
    )
    args = parser.parse_args()
    main(pathlib.Path(args.pandda_dataset_dir))