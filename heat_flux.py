from typing import Dict, Callable

from ase import Atoms
import jax.numpy as jnp
from jax import jacfwd
from jax_md.partition import NeighborList


def compute_naive(apply_fn: Callable, params, state: Dict, R: jnp.ndarray, Z: jnp.ndarray, neighbors: NeighborList,
                  atoms: Atoms) -> jnp.ndarray:
    jacobian_fn = jacfwd(apply_fn, argnums=2)
    jacobian, state = jacobian_fn(params, state, R, Z, neighbors)

    R_ref = atoms.get_all_distances(vector=True, mic=True)

    virials = jnp.zeros(shape=(96, 3, 3))
    for i, r in enumerate(R):
        jac = jacobian[i]  # (j, beta)
        r_ref = R_ref[i]  # (j, alpha)

        # virials += jnp.outer(r_ref, jac)
        virials += -r_ref[:, :, None] * jac[:, None, :]

    return virials
