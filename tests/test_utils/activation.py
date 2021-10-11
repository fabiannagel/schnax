from typing import Dict, Tuple
import jax.numpy as jnp
import numpy as np
import torch


def _dispatch_to_numpy(tensor: torch.tensor):
    return tensor.cpu().numpy()[0]

def get_embeddings(state: Dict):
    return state['SchNet']['embedding']

def get_embeddings(schnet_activations: Dict, schnax_activations: Dict):
    return schnet_activations['representation.embedding'][0].numpy(), \
           schnax_activations['SchNet']['embedding']

def get_distance_expansion(state: Dict):
    return state['SchNet/~/GaussianSmearing']['GaussianSmearing']

def get_distance_expansion(schnet_activations: Dict, schnax_activations: Dict):
    return schnet_activations['representation.distance_expansion'][0].numpy(), \
           schnax_activations['SchNet/~/GaussianSmearing']['GaussianSmearing']


def get_interaction_output(schnet_activations: Dict, schnax_activations: Dict, interaction_block_idx=0):
    k = 'representation.interactions.{}.dense'.format(interaction_block_idx)
    a_schnet = schnet_activations[k][0]

    k = 'SchNet/~/Interaction_{}'.format(interaction_block_idx)
    return state[k]['Output']


def get_cfconv_filters(state: Dict, interaction_block_idx=0):
    k = 'SchNet/~/Interaction_{}/~/CFConv/~/FilterNetwork'.format(interaction_block_idx)
    return state[k]['linear_0'], state[k]['linear_1']


def get_cutoff_network(state: Dict, interaction_block_idx=0):
    k = 'SchNet/~/Interaction_{}/~/CFConv/~/HardCutoff'.format(interaction_block_idx)
    return state[k]['HardCutoff']