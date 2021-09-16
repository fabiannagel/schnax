from ase.io import read
import numpy as np
import jax.numpy as jnp
from jax import jit
import jax_md
from jax_md import space
from jax_md.partition import NeighborList
from jax_md.space import DisplacementFn


def get_input(geometry_file: str):
    atoms = read(geometry_file, format="aims")
    R = jnp.float32(atoms.positions)
    box = jnp.float32(atoms.get_cell().array)
    return R, box


def compute_distances_iteratively(R: jnp.ndarray, neighbor_list: NeighborList, displacement_fn: DisplacementFn,
                                  r_cutoff: float, dr_threshold: float) -> jnp.ndarray:

    distances = []

    for r, neighbor_indices in zip(R, neighbor_list.idx):

        neighbor_distances = []
        neighbors_of_r = R[neighbor_indices]

        for neighbor_idx, neighbor in zip(neighbor_indices, neighbors_of_r):

            # if the index is padding, the distance is 0
            if neighbor_idx == R.shape[0]:
                neighbor_distances.append(0)
                continue

            displacements = displacement_fn(r, neighbor)
            distance = jnp.sqrt(jnp.sum(displacements ** 2))

            # yeah. not happening anymore.
            if distance > (r_cutoff + dr_threshold):
                print("{} > {}".format(distance, r_cutoff + dr_threshold))

            neighbor_distances.append(distance)

        distances.append(neighbor_distances)

    return jnp.array(distances)


def compute_distances_vectorized(R: jnp.ndarray, neighbor_list: NeighborList, displacement_fn: DisplacementFn) -> jnp.ndarray:
    R_neighbors = R[neighbor_list.idx]

    nl_displacement_fn = space.map_neighbor(displacement_fn)
    displacements = nl_displacement_fn(R, R_neighbors)
    distances_with_padding = space.distance(displacements)

    padding_mask = (neighbor_list.idx < R.shape[0])
    distances_without_padding = distances_with_padding * padding_mask
    return distances_without_padding


r_cutoff = 5.0
dr_threshold = 1.0
R, box = get_input("geometry.in")

displacement_fn, shift_fn = jax_md.space.periodic_general(box, fractional_coordinates=False)

neighbor_fn = jax_md.partition.neighbor_list(
    displacement_fn,
    box,
    r_cutoff,
    dr_threshold,
    capacity_multiplier=1.0,
    mask_self=False,
    disable_cell_list=True,
    fractional_coordinates=False)

neighbor_list = neighbor_fn(R)

distances_iteratively = compute_distances_iteratively(R, neighbor_list, displacement_fn, r_cutoff, dr_threshold)
distances_vectorized = compute_distances_vectorized(R, neighbor_list, displacement_fn)
np.testing.assert_allclose(distances_vectorized, distances_iteratively, atol=1e-6, rtol=1e-6)