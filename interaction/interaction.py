import haiku as hk
import jax.numpy as jnp
from jax_md.partition import NeighborList

from interaction.cfconv import CFConv
from utils import shifted_softplus


class Interaction(hk.Module):

    def __init__(self, n_atom_basis: int, n_filters: int, n_spatial_basis: int):
        super().__init__(name="Interaction")

        self.filter_network = hk.Sequential([
            hk.Linear(n_filters), shifted_softplus,     # n_spatial_basis -> n_filters
            hk.Linear(n_filters)                        # n_filters -> n_filters
        ])

        # TODO: cutoff network - do we need this?
        self.cutoff_network = None
        self.cfconv = CFConv(n_atom_basis, n_filters, n_atom_basis, self.filter_network, self.cutoff_network, activation=shifted_softplus)
        self.dense = hk.Linear(n_atom_basis)

    def __call__(self, x: jnp.ndarray, dR: jnp.ndarray, neighbors: NeighborList, pairwise_mask: jnp.ndarray, dR_expanded=None):
        """Compute convolution block.

                Args:
                    x: input representation/embedding of atomic environments with (N_a, n_in) shape.
                    dR: interatomic distances of (N_a, N_nbh) shape.
                    neighbors: neighbor list with neighbor indices in (N_a, N_nbh) shape.
                    pairwise_mask: mask to filter out non-existing neighbors introduced via padding.
                    dR_expanded (optional): expanded interatomic distances in a basis.
                        If None, dR.unsqueeze(-1) is used.

                Returns:
                    jnp.ndarray: block output with (N_a, n_out) shape.

            """
        x = self.cfconv(x, dR, neighbors, pairwise_mask, dR_expanded)
        x = self.dense(x)
        return x

