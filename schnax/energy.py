import haiku as hk
import jax_md
from jax import numpy as jnp
from jax_md.energy import DisplacementFn, Box

from .model import SchNet
from .utils import compute_distances


def _get_model(displacement_fn: DisplacementFn, n_atom_basis: int, max_z: int, n_gaussians: int, n_filters: int,
               mean: float, stddev: float, r_cutoff: jnp.ndarray, n_interactions: int, normalize_filter: bool,
               per_atom: bool, return_activations: bool):
    """Moved to dedicated method for better testing access."""

    # @hk.without_apply_rng
    # @hk.transform_with_state
    def model(R: jnp.ndarray, Z: jnp.int32, neighbors: jnp.ndarray):
        dR = compute_distances(R, neighbors, displacement_fn)
        net = SchNet(n_atom_basis=n_atom_basis,
                     max_z=max_z,
                     n_gaussians=n_gaussians,
                     n_filters=n_filters,
                     mean=mean,
                     stddev=stddev,
                     r_cutoff=r_cutoff,
                     n_interactions=n_interactions,
                     normalize_filter=normalize_filter,
                     per_atom=per_atom)
        return net(dR, Z, neighbors)

    if return_activations:
        model = hk.transform_with_state(model)

    return hk.without_apply_rng(model)


def schnet_neighbor_list(
        displacement_fn: DisplacementFn,
        box_size: Box,
        r_cutoff: float,
        dr_threshold: float,
        per_atom=False,
        n_interactions=1,
        n_atom_basis=128,
        max_z=100,
        n_gaussians=25,
        n_filters=128,
        mean=0.0,
        stddev=20.0,
        normalize_filter=False
):
    """Convenience wrapper around SchNet"""
    model = _get_model(displacement_fn=displacement_fn,
                       n_atom_basis=n_atom_basis,
                       max_z=max_z,
                       n_gaussians=n_gaussians,
                       n_filters=n_filters,
                       mean=mean,
                       stddev=stddev,
                       r_cutoff=r_cutoff,
                       n_interactions=n_interactions,
                       normalize_filter=normalize_filter,
                       per_atom=per_atom,
                       return_activations=True)

    neighbor_fn = jax_md.partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff,
        dr_threshold=dr_threshold,
        mask_self=True,
        fractional_coordinates=False,
    )

    return neighbor_fn, model.init, model.apply
