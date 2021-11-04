import jax.numpy as jnp

from jax_md import space
from jax_md.energy import DisplacementFn
from jax_md.partition import NeighborList


def compute_distances(
    R: jnp.ndarray, neighbors: NeighborList, displacement_fn: DisplacementFn
) -> jnp.ndarray:
    R_neighbors = R[neighbors.idx]

    nl_displacement_fn = space.map_neighbor(displacement_fn)
    displacements = nl_displacement_fn(R, R_neighbors)
    distances_with_padding = space.distance(displacements)

    padding_mask = neighbors.idx < R.shape[0]
    distances_without_padding = distances_with_padding * padding_mask
    return distances_without_padding
