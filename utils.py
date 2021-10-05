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


def get_params(torch_model_file: str) -> Dict:
    """Read weights from an existing SchNetPack torch model."""
    torch_model = torch.load(torch_model_file)['state']
    get_param = lambda k: torch_model[k].cpu().numpy()
    params = {}

    def set_params(layer_key: str, weight_key: str, bias_key=None):
        params[layer_key] = {}
        params[layer_key]['w'] = get_param(weight_key).T
        if bias_key:
            params[layer_key]['b'] = get_param(bias_key)

    # embeddings layer (special case, no transpose)
    params['SchNet/~/embeddings'] = {
        'embeddings': get_param('representation.embedding.weight')
    }

    # interaction block // cfconv block // filter network
    set_params(layer_key='SchNet/~/Interaction_0/~/CFConv/~/FilterNetwork/~/linear_0', weight_key='representation.interactions.0.filter_network.0.weight', bias_key='representation.interactions.0.filter_network.0.bias')
    set_params(layer_key='SchNet/~/Interaction_0/~/CFConv/~/FilterNetwork/~/linear_1', weight_key='representation.interactions.0.filter_network.1.weight', bias_key='representation.interactions.0.filter_network.1.bias')

    # interaction block // cfconv block // in2f
    set_params(layer_key='SchNet/~/Interaction_0/~/CFConv/~/in2f', weight_key='representation.interactions.0.cfconv.in2f.weight')

    # interaction block // cfconv block // f2out
    set_params(layer_key='SchNet/~/Interaction_0/~/CFConv/~/f2out', weight_key='representation.interactions.0.cfconv.f2out.weight', bias_key='representation.interactions.0.cfconv.f2out.bias')

    # interaction block // output layer
    set_params(layer_key='SchNet/~/Interaction_0/~/Output', weight_key='representation.interactions.0.dense.weight', bias_key='representation.interactions.0.dense.bias')

    return to_haiku_dict(params)


def get_input(geometry_file: str):
    atoms = read(geometry_file, format="aims")
    R = atoms.positions.astype(np.float32)
    Z = atoms.numbers.astype(np.int)
    box = np.array(atoms.cell.array, dtype=np.float32)
    return jnp.float32(R), jnp.int32(Z), jnp.float32(box)


def compute_distances(R: jnp.ndarray, neighbors: NeighborList, displacement_fn: DisplacementFn) -> jnp.ndarray:
    R_neighbors = R[neighbors.idx]

    nl_displacement_fn = space.map_neighbor(displacement_fn)
    displacements = nl_displacement_fn(R, R_neighbors)
    distances_with_padding = space.distance(displacements)

    padding_mask = (neighbors.idx < R.shape[0])
    distances_without_padding = distances_with_padding * padding_mask
    return distances_without_padding


def shifted_softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softplus(x) - jnp.log(2.0)