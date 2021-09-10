from typing import Dict, Tuple

import torch
from schnet.convert import get_converter
from ase.io import read
import jax.numpy as jnp


def get_input(geometry_file: str, r_cutoff: float) -> Tuple[jnp.float32, jnp.float32, jnp.float32]:
    converter = get_converter(r_cutoff, device="cpu")
    atoms = read(geometry_file, format="aims")
    inputs = converter(atoms)

    box = jnp.float32(inputs['_cell'][0].numpy())
    R = jnp.float32(inputs['_positions'][0].numpy())
    Z = jnp.float32(inputs['_atomic_numbers'][0].numpy())
    return R, Z, box


def get_params(torch_model_file: str) -> Dict:
    torch_model = torch.load(torch_model_file)

    list(filter(lambda k: k.startswith("representation"), torch_model['state'].keys()))
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
    # torch_model['fc1.weight'].cpu().numpy().T
    # torch_model['fc1.bias'].cpu().numpy()

    return params

# get_params("schnet/model_n1.torch")
