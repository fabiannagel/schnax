from collections import OrderedDict
from typing import Dict

import jax.nn
import jax.numpy as jnp
import numpy as np
import torch
from ase.io import read
from haiku._src.data_structures import to_haiku_dict
from jax import numpy as jnp
from jax_md import space
from jax_md.energy import DisplacementFn
from jax_md.partition import NeighborList


def compute_distances(R: jnp.ndarray, neighbors: NeighborList, displacement_fn: DisplacementFn) -> jnp.ndarray:
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

    #  'representation.interactions.0.filter_network.0.weight', 'representation.interactions.0.filter_network.0.bias',
    #  'representation.interactions.0.filter_network.1.weight', 'representation.interactions.0.filter_network.1.bias',
    #  'representation.interactions.0.cutoff_network.cutoff',
    #
    #  'representation.interactions.0.cfconv.in2f.weight',
    #  'representation.interactions.0.cfconv.f2out.weight', 'representation.interactions.0.cfconv.f2out.bias',

    # TODO: are these duplicates?
    #  'representation.interactions.0.cfconv.filter_network.0.weight', 'representation.interactions.0.cfconv.filter_network.0.bias',
    #  'representation.interactions.0.cfconv.filter_network.1.weight', 'representation.interactions.0.cfconv.filter_network.1.bias',
    #
    #  'representation.interactions.0.cfconv.cutoff_network.cutoff',
    #  'representation.interactions.0.dense.weight',
    #  'representation.interactions.0.dense.bias']

    get_pristine = lambda k: torch_model[k].cpu().numpy()
    get_transposed = lambda k: get_pristine(k).T
    params = {}

    params['SchNet/~/embeddings'] = {
        'embeddings': get_pristine('representation.embedding.weight')   # special case for embeddings (no tranpose)
    }

    # interaction: filter network, first layer
    params['SchNet/~/Interaction_0/~/FilterNetwork/~/linear_0'] = {
        'w': get_transposed('representation.interactions.0.filter_network.0.weight'),
        'b': get_pristine('representation.interactions.0.filter_network.0.bias')
    }

    # interaction: filter network, second layer
    params['SchNet/~/Interaction_0/~/FilterNetwork/~/linear_1'] = {
        'w': get_transposed('representation.interactions.0.filter_network.1.weight'),
        'b': get_pristine('representation.interactions.0.filter_network.1.bias')
    }

    # interaction: cfconv, in2f layer
    params['SchNet/~/Interaction_0/~/CFConv/~/in2f'] = {
        'w': get_transposed('representation.interactions.0.cfconv.in2f.weight')
    }

    # interaction: cfconv, f2out layer
    params['SchNet/~/Interaction_0/~/CFConv/~/f2out'] = {
        'w': get_transposed('representation.interactions.0.cfconv.f2out.weight'),
        'b': get_pristine('representation.interactions.0.cfconv.f2out.bias')
    }

    # # interaction: output layer
    params['SchNet/~/Interaction_0/~/Output'] = {
        'w': get_transposed('representation.interactions.0.dense.weight'),
        'b': get_pristine('representation.interactions.0.dense.bias')
    }

    return to_haiku_dict(params)


def get_input(geometry_file: str):
    atoms = read(geometry_file, format="aims")
    R = atoms.positions.astype(np.float32)
    Z = atoms.numbers.astype(np.int)
    box = np.array(atoms.cell.array, dtype=np.float32)
    return jnp.float32(R), jnp.int32(Z), jnp.float32(box)


def shifted_softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softplus(x) - jnp.log(2.0)