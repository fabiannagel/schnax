import haiku as hk
from jax import numpy as jnp

from schnax.model.misc import shifted_softplus


class FilterNetwork(hk.Module):
    def __init__(self, n_filters: int):
        super().__init__(name="FilterNetwork")

        self.linear_0 = hk.Sequential(
            [hk.Linear(n_filters, name="linear_0"), shifted_softplus]
        )  # n_spatial_basis -> n_filters
        self.linear_1 = hk.Linear(n_filters, name="linear_1")  # n_filters -> n_filters

    def __call__(self, x: jnp.ndarray):
        x = self.linear_0(x)
        hk.set_state(
            self.linear_0.layers[0].name, x
        )  # w/o referencing layer 0, the key would be "Sequential".

        x = self.linear_1(x)
        hk.set_state(self.linear_1.name, x)

        return x
