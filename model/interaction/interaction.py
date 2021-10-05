import haiku as hk
import jax.numpy as jnp
from jax_md.partition import NeighborList

from model.interaction.cfconv import CFConv
from utils import shifted_softplus


class Interaction(hk.Module):

    def __init__(self, idx: int, n_atom_basis: int, n_filters: int, n_spatial_basis: int, r_cutoff: float):
        super().__init__(name="Interaction_{}".format(idx))
        self.cfconv = CFConv(n_filters, n_atom_basis, r_cutoff, activation=shifted_softplus)
        self.dense = hk.Linear(n_atom_basis, name="Output")

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
        hk.set_state(self.dense.name, x)
        return x

