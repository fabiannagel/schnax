from typing import Dict, Callable

from ase import Atoms
import jax.numpy as jnp
from jax import jacfwd, vmap
from jax.interpreters.xla import DeviceArray
from jax_md import space
from jax_md.partition import NeighborList


def compute_naive(apply_fn: Callable, params, state: Dict, R: jnp.ndarray, Z: jnp.ndarray, neighbors: NeighborList,
                  atoms: Atoms) -> jnp.ndarray:
    jacobian_fn = jacfwd(apply_fn, argnums=2)
    jacobian, state = jacobian_fn(params, state, R, Z, neighbors)

    # TODO: R_ref as argument so that we can compute it in JAX
    R_ref = atoms.get_all_distances(vector=True, mic=True)

    virials = jnp.zeros(shape=(96, 3, 3))
    for i, r in enumerate(R):
        jac = jacobian[i]  # (j, beta)      (96, 3)
        r_ref = R_ref[i]  # (j, alpha)      (96, 3)

        # adapt shapes to achieve outer product via hadamard
        r_ref = -r_ref[:, :, None]          # (96, 3, 1)
        jac = jac[:, None, :]               # (96, 1, 3)
        outer_product = r_ref * jac         # (96, 3, 3)

        virials += outer_product

    return virials


def compute_einsum(apply_fn: Callable, params, state: Dict, R: jnp.ndarray, Z: jnp.ndarray, neighbors: NeighborList,
                  atoms: Atoms) -> jnp.ndarray:
    jacobian_fn = jacfwd(apply_fn, argnums=2)
    jacobian, state = jacobian_fn(params, state, R, Z, neighbors)
    R_ref = atoms.get_all_distances(vector=True, mic=True)
    return jnp.einsum("i...a,i...b", -R_ref, jacobian)


def compute_vmapped(apply_fn: Callable, params, state: Dict, R: jnp.ndarray, Z: jnp.ndarray, neighbors: NeighborList,
                  atoms: Atoms):
    jacobian_fn = jacfwd(apply_fn, argnums=2)
    jacobian, state = jacobian_fn(params, state, R, Z, neighbors)

    R_ref = atoms.get_all_distances(vector=True, mic=True)






