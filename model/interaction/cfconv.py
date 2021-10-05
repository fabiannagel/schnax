from typing import Callable

import haiku as hk
import jax.numpy as jnp
from jax_md.partition import NeighborList

from model.interaction.aggregate import Aggregate


class CFConv(hk.Module):

    def __init__(self, n_in: int, n_filters: int, n_out: int, filter_network: Callable, cutoff_network: Callable,
                 activation, normalize_filter=False, axis=1):
        super().__init__(name="CFConv")

        self.in2f = hk.Linear(n_filters, with_bias=False, name="in2f")
        self.f2out = hk.Sequential([
            hk.Linear(n_out, with_bias=True, name="f2out"), activation
        ])
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        self.aggregate = Aggregate(axis=axis, mean_pooling=normalize_filter)

    def _reshape_y(self, y: jnp.ndarray, neighbors: NeighborList) -> jnp.ndarray:
        nbh_size = neighbors.idx.shape

        # (n_atoms, max_occupancy) -> (n_atoms * max_occupancy, 1)
        # for batches, use shape (-1, nbh_size[1] * nbh_size[2], 1)
        nbh = neighbors.idx.reshape((nbh_size[0] * nbh_size[1], 1))

        # (n_atoms * max_occupancy, 1) -> (n_atoms * max_occupancy, n_filters)
        nbh = jnp.tile(nbh, (1, y.shape[1]))

        # (n_atoms, n_filters) -> (n_atoms * max_occupancy, n_filters)
        y = jnp.take_along_axis(y, indices=nbh, axis=0)

        # (n_atoms * max_occupancy, n_filters) -> (n_atoms, max_occupancy, n_filters)
        y = jnp.reshape(y, (nbh_size[0], nbh_size[1], -1))
        return y

    def __call__(self, x: jnp.ndarray, dR: jnp.ndarray, neighbors: NeighborList, pairwise_mask: jnp.ndarray,
                 dR_expanded: jnp.ndarray):
        if dR_expanded is None:
            # Insert a new dimension (size 1) at the last position
            # (n_atoms, max_occupancy) -> (n_atoms, max_occupancy, 1)
            dR_expanded = jnp.expand_dims(dR, axis=-1)

        # pass expanded interactomic distances through filter block
        W = self.filter_network(dR_expanded)

        # apply cutoff
        if self.cutoff_network is not None:
            C = self.cutoff_network(dR)
            W = W * jnp.expand_dims(C, axis=-1)

        # pass initial embeddings through dense layer. reshape y for element-wise multiplication by W.
        y = self.in2f(x)
        y = self._reshape_y(y, neighbors)

        # element-wise multiplication, aggregation and dense output layer.
        y = y * W
        y = self.aggregate(y, pairwise_mask)
        y = self.f2out(y)
        return y
