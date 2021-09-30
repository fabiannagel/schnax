from typing import Dict

import jax.numpy as jnp
import numpy as np
import torch
from ase.io import read
from haiku._src.data_structures import to_haiku_dict
from jax_md import space
from jax_md.energy import DisplacementFn
from jax_md.partition import NeighborList


def compute_distances_vectorized(R: jnp.ndarray, neighbors: NeighborList, displacement_fn: DisplacementFn) -> jnp.ndarray:
    R_neighbors = R[neighbors.idx]

    nl_displacement_fn = space.map_neighbor(displacement_fn)
    displacements = nl_displacement_fn(R, R_neighbors)
    distances_with_padding = space.distance(displacements)

    padding_mask = (neighbors.idx < R.shape[0])
    distances_without_padding = distances_with_padding * padding_mask
    return distances_without_padding


def get_params(torch_model_file: str) -> Dict:
    torch_model = torch.load(torch_model_file)
    torch_model = torch_model['state']

    # list(filter(lambda k: k.startswith("representation"), torch_model.keys()))
    # ['representation.embedding.weight',
    #  'representation.distance_expansion.width', 'representation.distance_expansion.offsets',

    #  'representation.interactions.0.filter_network.0.weight',
    #  'representation.interactions.0.filter_network.0.bias', 'representation.interactions.0.filter_network.1.weight',
    #  'representation.interactions.0.filter_network.1.bias', 'representation.interactions.0.cutoff_network.cutoff',
    #  'representation.interactions.0.cfconv.in2f.weight', 'representation.interactions.0.cfconv.f2out.weight',
    #  'representation.interactions.0.cfconv.f2out.bias', 'representation.interactions.0.cfconv.filter_network.0.weight',
    #  'representation.interactions.0.cfconv.filter_network.0.bias',
    #  'representation.interactions.0.cfconv.filter_network.1.weight',
    #  'representation.interactions.0.cfconv.filter_network.1.bias',
    #  'representation.interactions.0.cfconv.cutoff_network.cutoff', 'representation.interactions.0.dense.weight',
    #  'representation.interactions.0.dense.bias']

    params = {}
    params['SchNet/~/embeddings'] = {}
    params['SchNet/~/embeddings']['embeddings'] = torch_model['representation.embedding.weight'].cpu().numpy()

    return to_haiku_dict(params)


def get_input(geometry_file: str):
    atoms = read(geometry_file, format="aims")
    R = atoms.positions.astype(np.float32)
    Z = atoms.numbers.astype(np.int)
    box = np.array(atoms.cell.array, dtype=np.float32)
    return jnp.float32(R), jnp.int32(Z), jnp.float32(box)