from typing import Dict, Tuple
import jax.numpy as jnp
import numpy as np
import torch


def _dispatch_to_numpy(tensor: torch.tensor):
    return tensor.cpu().numpy()[0]


def get_embeddings(schnet_activations: Dict, schnax_activations: Dict):
    return schnet_activations['representation.embedding'][0].numpy(), \
           schnax_activations['SchNet']['embedding']


def get_distance_expansion(schnet_activations: Dict, schnax_activations: Dict):
    return schnet_activations['representation.distance_expansion'][0].numpy(), \
           schnax_activations['SchNet/~/GaussianSmearing']['GaussianSmearing']


def get_interaction_output(schnet_activations: Dict, schnax_activations: Dict, interaction_block_idx=0):
    k = 'representation.interactions.{}.dense'.format(interaction_block_idx)
    a_schnet = schnet_activations[k][0]

    k = 'SchNet/~/Interaction_{}'.format(interaction_block_idx)
    a_schnax = schnax_activations[k]['Output']
    return a_schnax, a_schnet


def get_cfconv_filters(schnet_activations: Dict, schnax_activations: Dict, interaction_block_idx=0):
    k = 'representation.interactions.{}.filter_network.1'.format(interaction_block_idx)
    a_schnet_1 = _dispatch_to_numpy(schnet_activations[k])

    k = 'SchNet/~/Interaction_{}/~/CFConv/~/FilterNetwork'.format(interaction_block_idx)
    a_schnax_1 = schnax_activations[k]['linear_1']

    return a_schnet_1, a_schnax_1


def get_cutoff_network(schnet_activations: Dict, schnax_activations: Dict, interaction_block_idx=0):
    k = 'representation.interactions.{}.cutoff_network'.format(interaction_block_idx)
    schnet_cutoff = _dispatch_to_numpy(schnet_activations[k])

    k = 'SchNet/~/Interaction_{}/~/CFConv/~/HardCutoff'.format(interaction_block_idx)
    schnax_cutoff = schnax_activations[k]['HardCutoff']

    return schnet_cutoff, schnax_cutoff

