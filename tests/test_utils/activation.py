from typing import Dict, Tuple
import jax.numpy as jnp
import numpy as np
import torch


def _dispatch_to_numpy(tensor: torch.tensor):
    return tensor.cpu().numpy()[0]


def get_embeddings(schnet_activations: Dict, schnax_activations: Dict):
    return (
        schnet_activations['representation.embedding'][0].numpy(),
        schnax_activations['SchNet']['embedding'],
    )


def get_distance_expansion(schnet_activations: Dict, schnax_activations: Dict):
    return (
        schnet_activations['representation.distance_expansion'][0].numpy(),
        schnax_activations['SchNet/~/GaussianSmearing']['GaussianSmearing'],
    )


def get_interaction_output(
    schnet_activations: Dict, schnax_activations: Dict, interaction_block_idx: int
):
    k = 'representation.interactions.{}.dense'.format(interaction_block_idx)
    a_schnet = schnet_activations[k][0]

    k = 'SchNet/~/Interaction_{}'.format(interaction_block_idx)
    a_schnax = schnax_activations[k]['Output']
    return a_schnax, a_schnet


def get_cfconv_filters(
    schnet_activations: Dict, schnax_activations: Dict, interaction_block_idx: int
):
    k = 'representation.interactions.{}.filter_network.1'.format(interaction_block_idx)
    a_schnet_1 = _dispatch_to_numpy(schnet_activations[k])

    k = 'SchNet/~/Interaction_{}/~/CFConv/~/FilterNetwork'.format(interaction_block_idx)
    a_schnax_1 = schnax_activations[k]['linear_1']

    return a_schnet_1, a_schnax_1


def get_cutoff_network(
    schnet_activations: Dict, schnax_activations: Dict, interaction_block_idx: int
):
    k = 'representation.interactions.{}.cutoff_network'.format(interaction_block_idx)
    schnet_cutoff = _dispatch_to_numpy(schnet_activations[k])

    k = 'SchNet/~/Interaction_{}/~/CFConv/~/CosineCutoff'.format(interaction_block_idx)
    schnax_cutoff = schnax_activations[k]['CosineCutoff']

    return schnet_cutoff, schnax_cutoff


def get_in2f(schnet_activations: Dict, schnax_activations: Dict, interaction_block_idx: int):
    k = 'representation.interactions.{}.cfconv.in2f'.format(interaction_block_idx)
    schnet_in2f = _dispatch_to_numpy(schnet_activations[k])

    k = 'SchNet/~/Interaction_{}/~/CFConv'.format(interaction_block_idx)
    schnax_in2f = schnax_activations[k]['in2f']

    return schnet_in2f, schnax_in2f


def get_aggregate(
    schnet_activations: Dict, schnax_activations: Dict, interaction_block_idx: int
):
    k = 'representation.interactions.{}.cfconv.agg'.format(interaction_block_idx)
    schnet_agg = _dispatch_to_numpy(schnet_activations[k])

    k = 'SchNet/~/Interaction_{}/~/CFConv'.format(interaction_block_idx)
    schnax_agg = schnax_activations[k]['Aggregate']

    return schnet_agg, schnax_agg


def get_f2out(
    schnet_activations: Dict, schnax_activations: Dict, interaction_block_idx: int
):
    k = 'representation.interactions.{}.cfconv.f2out'.format(interaction_block_idx)
    schnet_f2out = _dispatch_to_numpy(schnet_activations[k])

    k = 'SchNet/~/Interaction_{}/~/CFConv'.format(interaction_block_idx)
    schnax_f2out = schnax_activations[k]['f2out']

    return schnet_f2out, schnax_f2out
