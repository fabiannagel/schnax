from functools import partial
from typing import Dict, Tuple

import torch
from haiku._src.data_structures import to_haiku_dict
from jax_md.energy import DisplacementFn
from jax_md.partition import NeighborList

from schnet.convert import get_converter
from ase.io import read
import jax
import jax.numpy as jnp
import jax_md


def compute_nl_distances(displacement_fn: DisplacementFn, R: jnp.ndarray, neighbors: NeighborList, dimension_wise=False, **kwargs):
    """Compute interatomic distances for a matrix of atomic distances and their neighbor list indices."""
    d = partial(displacement_fn, **kwargs)
    d = jax_md.space.map_neighbor(d)

    R_neighbors = R[neighbors.idx]
    dR = d(R, R_neighbors)

    if dimension_wise:
        return dR

    # reduce dimension-wise distances to vector magnitude
    magnitude_fn = lambda x: jnp.sqrt(jnp.sum(x ** 2))
    vectorized_fn = jax.vmap(jax.vmap(magnitude_fn, in_axes=0), in_axes=0)
    dR_magnitudes = vectorized_fn(dR)
    return dR_magnitudes


def get_input(geometry_file: str, r_cutoff: float) -> Tuple[jnp.float32, jnp.float32, jnp.float32]:
    converter = get_converter(r_cutoff, device="cpu")
    atoms = read(geometry_file, format="aims")
    inputs = converter(atoms)

    box = jnp.float32(inputs['_cell'][0].numpy())
    R = jnp.float32(inputs['_positions'][0].numpy())
    Z = jnp.int32(inputs['_atomic_numbers'][0].numpy())
    return R, Z, box


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

# get_params("schnet/model_n1.torch")
