from typing import Dict


def get_embeddings(state: Dict):
    return state['SchNet']['embedding']


def get_distance_expansion(state: Dict):
    return state['SchNet/~/GaussianSmearing']['GaussianSmearing']


def get_interaction_output(state: Dict, interaction_block_idx=0):
    k = 'SchNet/~/Interaction_{}'.format(interaction_block_idx)
    return state[k]['Output']


def get_cfconv_filters(state: Dict, interaction_block_idx=0):
    k = 'SchNet/~/Interaction_{}/~/CFConv/~/FilterNetwork'.format(interaction_block_idx)
    return state[k]['linear_0'], state[k]['linear_1']


def get_cutoff_network(state: Dict, interaction_block_idx=0):
    k = 'SchNet/~/Interaction_{}/~/CFConv/~/HardCutoff'.format(interaction_block_idx)
    return state[k]['HardCutoff']