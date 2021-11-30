from typing import Dict, Any

import numpy as np
from jax import random
from jax_md import space, partition
from schnax import energy, utils
from schnetkit.engine import load_file
import jax.numpy as jnp
import regex as re


def get_interaction_count(file: str) -> int:
    spec, state = load_file(file)
    return spec['schnet']['representation']['n_interactions']


def get_params(file: str) -> Dict:
    """Read weights from an existing schnetkit torch model."""
    spec, state = load_file(file)
    n_interactions = get_interaction_count(file)
    params = {}

    def get_param(key):
        return state[key].cpu().numpy()

    def set_params(layer_key: str, weight_key: str, bias_key=None, interaction_idx=None):
        if interaction_idx is not None:
            layer_key = layer_key.format(interaction_idx)
            weight_key = weight_key.format(interaction_idx)
            if bias_key:
                bias_key = bias_key.format(interaction_idx)

        params[layer_key] = {}
        params[layer_key]['w'] = get_param(weight_key).T
        if bias_key:
            params[layer_key]['b'] = get_param(bias_key)

    # embeddings layer (special case, no transpose)
    params['SchNet/~/embeddings'] = {
        'embeddings': get_param('representation.embedding.weight')
    }

    for i in range(n_interactions):
        # interaction block // cfconv block // filter network
        set_params(
            layer_key='SchNet/~/Interaction_{}/~/CFConv/~/FilterNetwork/~/linear_0',
            weight_key='representation.interactions.{}.filter_network.0.weight',
            bias_key='representation.interactions.{}.filter_network.0.bias',
            interaction_idx=i,
        )

        set_params(
            layer_key='SchNet/~/Interaction_{}/~/CFConv/~/FilterNetwork/~/linear_1',
            weight_key='representation.interactions.{}.filter_network.1.weight',
            bias_key='representation.interactions.{}.filter_network.1.bias',
            interaction_idx=i,
        )

        # interaction block // cfconv block // in2f
        set_params(
            layer_key='SchNet/~/Interaction_{}/~/CFConv/~/in2f',
            weight_key='representation.interactions.{}.cfconv.in2f.weight',
            interaction_idx=i,
        )

        # interaction block // cfconv block // f2out
        set_params(
            layer_key='SchNet/~/Interaction_{}/~/CFConv/~/f2out',
            weight_key='representation.interactions.{}.cfconv.f2out.weight',
            bias_key='representation.interactions.{}.cfconv.f2out.bias',
            interaction_idx=i,
        )

        # interaction block // output layer
        set_params(
            layer_key='SchNet/~/Interaction_{}/~/Output',
            weight_key='representation.interactions.{}.dense.weight',
            bias_key='representation.interactions.{}.dense.bias',
            interaction_idx=i,
        )

    set_params(
        layer_key='SchNet/~/atomwise/~/linear_0',
        weight_key='output_modules.0.out_net.1.out_net.0.weight',
        bias_key='output_modules.0.out_net.1.out_net.0.bias',
    )
    set_params(
        layer_key='SchNet/~/atomwise/~/linear_1',
        weight_key='output_modules.0.out_net.1.out_net.1.weight',
        bias_key='output_modules.0.out_net.1.out_net.1.bias',
    )

    return params


def normalize_representation_config(repr_config: Dict) -> Dict:
    """Normalize existing representation config to contain all required arguments. Also changed some naming conventions
    to passthrough kwargs to schnax with less hassle."""

    normalized_repr = {}

    def set_value_or_default(key: str, default: Any, new_key=None):
        if new_key is None:
            new_key = key

        try:
            normalized_repr.update({new_key: repr_config[key]})
        except KeyError:
            normalized_repr.update({new_key: default})

    # skipping 'trainable_gaussians' (training not implemented)
    set_value_or_default('cutoff', 5.0, new_key='r_cutoff')
    set_value_or_default('n_interactions', 1)
    set_value_or_default('n_atom_basis', 128)
    set_value_or_default('max_z', 100)
    set_value_or_default('n_gaussians', 25)
    set_value_or_default('n_filters', 128)
    set_value_or_default('mean', 0.0)
    set_value_or_default('stddev', 1.0)
    set_value_or_default('normalize_filter', False)
    return normalized_repr


def initialize_from_schnetkit_model(
    file: str, box: np.ndarray, dr_threshold=0.0, per_atom=False, return_activations=True
):
    spec, weights = load_file(file)
    model_config = normalize_representation_config(spec['schnet']['representation'])
    atomwise_config = spec['schnet']['atomwise']

    # these keys are in atomwise
    model_config['mean'] = atomwise_config['mean']
    model_config['stddev'] = atomwise_config['stddev']

    box = jnp.float32(box)
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)

    r_cutoff = jnp.float32(model_config['r_cutoff'])
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box,
        r_cutoff,
        dr_threshold=dr_threshold,  # as the effective cutoff = r_cutoff + dr_threshold
        mask_self=True,  # an atom is not a neighbor of itself
        fractional_coordinates=False,
    )

    # 'max_z', 'n_gaussians', 'mean', and 'stddev' missing in repr
    init_fn, apply_fn = energy._get_model(
        displacement_fn=displacement_fn, per_atom=per_atom, return_activations=return_activations, **model_config
    )

    params = utils.get_params(file)
    return neighbor_fn, displacement_fn, shift_fn, params, init_fn, apply_fn
