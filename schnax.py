from functools import partial
from typing import OrderedDict, List, Dict

import torch
import haiku as hk
from jax_md.energy import DisplacementFn, Box
from schnetpack import Properties

import utils
import jax_md
import numpy as np
import jax.numpy as jnp
import jax

class Schnax(hk.Module):
    n_atom_basis = 128
    max_z = 100

    def __init__(self):
        super().__init__(name="SchNet")

        self.embedding = hk.Embed(self.max_z, self.n_atom_basis, name="embeddings")       # TODO: Torch padding_idx missing in Haiku.
        # distances: let JAX-MD handle this
        # TODO: function for distance expansion
        # TODO: interactions blocks

    def __call__(self, dR: jnp.ndarray, Z: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        # get embedding for Z
        x = self.embedding(Z)
        # expand interatomic distances (for example, Gaussian smearing)
        # compute interactions
        return x


"""Convenience wrapper around Schnax"""
def schnet_neighbor_list(displacement_fn: DisplacementFn,
                         box_size: Box,
                         r_cutoff: float,
                         dr_threshold: float,
                         params: Dict):

    @hk.without_apply_rng
    @hk.transform
    def model(R: jnp.ndarray, Z: jnp.int32, neighbors: jnp.ndarray, **kwargs):
        dR = utils.compute_neighbor_list_distances(displacement_fn, R, neighbors)
        net = Schnax()
        return net(dR, Z)

    neighbor_fn = jax_md.partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff,
        dr_threshold,
        capacity_multiplier=1.0,
        mask_self=False,
        fractional_coordinates=False)

    init_fn, apply_fn = model.init, model.apply
    return neighbor_fn, init_fn, apply_fn


def predict(geometry_file: str):
    r_cutoff = 5.0
    dr_threshold = 1.0  # TODO: Does this make sense?
    R, Z, box = utils.get_input(geometry_file, r_cutoff)
    params = utils.get_params("schnet/model_n1.torch")

    displacement_fn, shift_fn = jax_md.space.periodic_general(box, fractional_coordinates=False)
    neighbor_fn, init_fn, apply_fn = schnet_neighbor_list(displacement_fn, box, r_cutoff, dr_threshold, params)

    # compute neighbor list
    neighbors = neighbor_fn(R)

    # obtain PRNG key
    rng = jax.random.PRNGKey(0)

    # initialize model with a single example position and charge
    # we won't need these params as we will load the PyTorch model instead.
    _ = init_fn(rng, R, Z, neighbors)

    # print(init_params)
    pred = apply_fn(params, R, Z, neighbors)
    return pred


energy = predict("schnet/geometry.in")




