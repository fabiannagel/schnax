import haiku as hk
from jax import numpy as jnp

from utils import shifted_softplus


class FilterNetwork(hk.Module):

    def __init__(self, n_filters: int):
        super().__init__(name="FilterNetwork")

        self.network = hk.Sequential([
            hk.Linear(n_filters, name="linear_0"), shifted_softplus,  # n_spatial_basis -> n_filters
            hk.Linear(n_filters, name="linear_1")  # n_filters -> n_filters
        ])

    def __call__(self, x: jnp.ndarray):
        return self.network(x)