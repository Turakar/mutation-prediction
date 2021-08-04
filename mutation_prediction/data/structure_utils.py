import numpy as np
from Bio import pairwise2
from numba import jit

import mutation_prediction.data as data
from mutation_prediction.data import Dataset


def get_index_maps(dataset: Dataset):
    # tested manually via debugger inspection because good unit tests are difficult here

    # base sequence to string
    base_sequence_array = dataset.get_sequence()
    base_sequence = ""
    for a in base_sequence_array:
        base_sequence += data.index_to_acid(a)

    # build residue to structure sequence mapping
    structure = dataset.get_structure()
    residues = structure[0]["A"].get_list()
    structure_sequence = ""
    residue_to_structure_sequence = {}
    structure_sequence_to_residue = {}
    for i, r in enumerate(residues):
        shortname = r.get_resname()
        shortname = shortname[0] + shortname[1:].lower()
        try:
            acid = data.acid_shortname_to_letter(shortname)
            structure_sequence += acid
            residue_to_structure_sequence[i] = len(structure_sequence) - 1
            structure_sequence_to_residue[len(structure_sequence) - 1] = i
        except KeyError:
            residue_to_structure_sequence[i] = -1
            pass

    # build structure sequence to base sequence mapping
    alignment = pairwise2.align.globalxx(base_sequence, structure_sequence)[0]
    base_index = 0
    structure_index = 0
    base_to_structure_sequence = {}
    structure_to_base_sequence = {}
    for a, b in zip(alignment.seqA, alignment.seqB):
        if a == "-":
            structure_to_base_sequence[structure_index] = -1
            structure_index += 1
        elif b == "-":
            base_to_structure_sequence[base_index] = -1
            base_index += 1
        else:
            base_to_structure_sequence[base_index] = structure_index
            structure_to_base_sequence[structure_index] = base_index
            base_index += 1
            structure_index += 1

    # chain both mappings
    residue_to_base_sequence = {}
    for k, v in residue_to_structure_sequence.items():
        if v == -1:
            residue_to_base_sequence[k] = -1
        else:
            vv = structure_to_base_sequence[v]
            if vv == -1:
                residue_to_base_sequence[k] = -1
            else:
                residue_to_base_sequence[k] = vv
    base_to_residue_sequence = {}
    for k, v in base_to_structure_sequence.items():
        if v == -1:
            base_to_residue_sequence[k] = -1
        else:
            vv = structure_sequence_to_residue[v]  # cannot fail
            base_to_residue_sequence[k] = vv
    return base_to_residue_sequence, residue_to_base_sequence


def get_contacts(dataset: Dataset, max_distance: float = 8.0):
    # tested manually via debugger inspection because good unit tests are difficult here
    base_to_residue, _ = get_index_maps(dataset)
    contacts = []
    sequence = dataset.get_sequence()
    residues = dataset.get_structure()[0]["A"].get_list()
    for i in range(len(sequence)):
        if base_to_residue[i] == -1:
            continue
        residue_i = residues[base_to_residue[i]]
        for j in range(i + 1, len(sequence)):
            if base_to_residue[j] == -1:
                continue
            residue_j = residues[base_to_residue[j]]
            try:
                # CA = C-alpha atoms
                distance = residue_i["CA"] - residue_j["CA"]
                if distance <= max_distance:
                    contacts.append((i, j))
            except KeyError:
                pass
    return np.asarray(contacts)


def prepare_gridify(dataset: Dataset, grid_size: int):
    # preparation
    positions = get_positions(dataset)
    sequence_length = len(dataset.get_sequence())

    # calculate grid coords per acid
    all_grid_coords = np.zeros((sequence_length, 3), dtype=np.int32)
    min_coords = np.nanmin(positions, axis=0)
    coords_ranges = np.nanmax(positions, axis=0) - np.nanmin(positions, axis=0)
    coords_range = np.max(coords_ranges)
    for i, p in enumerate(positions):
        if not np.any(np.isnan(p)):
            p_normalized = (p - min_coords + (coords_range - coords_ranges) / 2) / coords_range
            grid_coords = np.floor(p_normalized * grid_size).astype(np.uint32)
            grid_coords[grid_coords == grid_size] = grid_size - 1
            all_grid_coords[i] = grid_coords
        else:
            all_grid_coords[i] = -1

    return all_grid_coords, grid_size


def gridify(dataset: Dataset, index: int, prep_values) -> np.ndarray:
    all_grid_coords, grid_size = prep_values

    # obtain mutated sequence
    num_mutations = dataset.get_num_mutations()[index]
    sequence = dataset.get_sequence().copy()
    sequence[dataset.get_positions()[index, :num_mutations]] = dataset.get_acids()[
        index, :num_mutations
    ]

    # apply to grid
    return _apply_to_grid(grid_size, data.num_acids(), all_grid_coords, sequence)


@jit(nopython=True)
def _apply_to_grid(
    grid_size: int, num_acids: int, all_grid_coords: np.ndarray, sequence: np.ndarray
) -> np.ndarray:
    grid = np.zeros((grid_size, grid_size, grid_size, num_acids), dtype=np.int32)
    for grid_coords, acid in zip(all_grid_coords, sequence):
        if grid_coords[0] >= 0:
            grid[grid_coords[0], grid_coords[1], grid_coords[2], acid] += 1

    return grid


def get_positions(dataset: Dataset) -> np.ndarray:
    _, residue_to_base = get_index_maps(dataset)
    residues = dataset.get_structure()[0]["A"]
    sequence_length = len(dataset.get_sequence())
    positions = np.zeros((sequence_length, 3), dtype=np.float32)
    positions.fill(np.nan)
    for i, residue in enumerate(residues):
        if residue_to_base[i] == -1:
            continue
        positions[residue_to_base[i]] = residue["CA"].get_coord()
    return positions


def knn(dataset: Dataset, k: int, indices_masked: bool = False):
    positions = get_positions(dataset)
    mask = np.logical_not(np.any(np.isnan(positions), axis=1))
    index_map = np.arange(len(positions))[mask]
    eff_positions = positions[mask]
    eff_len = len(eff_positions)
    nearest = np.zeros((eff_len, k), dtype=np.int32)
    distances = np.zeros_like(nearest, dtype=positions.dtype)
    distances[:] = np.inf
    for i, p in enumerate(eff_positions):
        for j, q in enumerate(eff_positions):
            d = np.linalg.norm(p - q)
            max_index = np.argmax(distances[i])
            if d < distances[i, max_index]:
                if indices_masked:
                    nearest[i, max_index] = j
                else:
                    nearest[i, max_index] = index_map[j]
                distances[i, max_index] = d
    indices = np.argsort(distances, axis=1)
    nearest = np.take_along_axis(nearest, indices, axis=1)
    distances = np.take_along_axis(distances, indices, axis=1)
    assert np.all(distances < np.inf)
    return nearest, distances, mask
