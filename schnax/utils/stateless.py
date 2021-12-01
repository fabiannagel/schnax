from typing import Callable
import jax.numpy as jnp
from haiku._src.data_structures import FlatMapping
from jax._src.random import KeyArray
from jax_md.partition import NeighborList


def transform_stateless(rng: KeyArray, init_fn: Callable, apply_fn: Callable):

    def stateless_init_fn(rng: KeyArray, R: jnp.ndarray, Z: jnp.ndarray, neighbors: NeighborList):
        params, state = init_fn(rng, R, Z, neighbors)
        return params

    def stateless_apply_fn(params: FlatMapping, R: jnp.ndarray, Z: jnp.ndarray, neighbors: NeighborList):
        _, state = init_fn(rng, R, Z, neighbors)

        pred, state = apply_fn(params, state, R, Z, neighbors)
        return pred

    return stateless_init_fn, stateless_apply_fn