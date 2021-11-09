import haiku as hk
import jax_md
from jax import numpy as jnp
from jax_md.energy import DisplacementFn, Box

from .model import SchNet
from .utils import compute_distances


def _get_model(displacement_fn: DisplacementFn, r_cutoff: float, n_interactions: int, per_atom=False):
    """Moved to dedicated method for better testing access."""

    @hk.without_apply_rng
    @hk.transform_with_state
    def model(R: jnp.ndarray, Z: jnp.int32, neighbors: jnp.ndarray):
        dR = compute_distances(R, neighbors, displacement_fn)
        net = SchNet(r_cutoff, n_interactions, per_atom)
        return net(dR, Z, neighbors)

    return model


def schnet_neighbor_list(
    displacement_fn: DisplacementFn,
    box_size: Box,
    r_cutoff: float,
    dr_threshold: float,
    n_interactions=1,
    per_atom=False,
):
    """Convenience wrapper around SchNet"""
    model = _get_model(displacement_fn, r_cutoff, n_interactions, per_atom)

    neighbor_fn = jax_md.partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff,
        dr_threshold=dr_threshold,
        mask_self=True,
        fractional_coordinates=False,
    )

    return neighbor_fn, model.init, model.apply
