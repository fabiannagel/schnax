import haiku as hk
from jax_md.partition import NeighborList

import jax.numpy as jnp

from model.gaussian_smearing import GaussianSmearing
from model.interaction.interaction import Interaction


class Schnax(hk.Module):
    n_atom_basis = 128
    max_z = 100
    n_gaussians = 25

    n_filters = 128
    n_interactions = 1

    def __init__(self, r_cutoff: float):
        super().__init__(name="SchNet")
        self.embedding = hk.Embed(self.max_z, self.n_atom_basis, name="embeddings")  # TODO: Torch padding_idx missing in Haiku.
        self.distance_expansion = GaussianSmearing(0.0, r_cutoff, self.n_gaussians)

        self.interactions = hk.Sequential([
            Interaction(idx=i, n_atom_basis=self.n_atom_basis,
                        n_filters=self.n_filters, n_spatial_basis=self.n_gaussians,
                        r_cutoff=r_cutoff) for i in range(self.n_interactions)
        ])


    def __call__(self, dR: jnp.ndarray, Z: jnp.ndarray, neighbors: NeighborList, *args, **kwargs) -> jnp.ndarray:
        # TODO: Move hk.set_state() calls into layer modules. Use self.name as key.

        # get embedding for Z
        x = self.embedding(Z)
        hk.set_state("embedding", x)

        # expand interatomic distances
        dR_expanded = self.distance_expansion(dR)
        # hk.set_state("distance_expansion", dR_expanded)

        # compute interactions
        pairwise_mask = None  # TODO: Figure out what this is

        for i, interaction in enumerate(self.interactions.layers):
            v = interaction(x, dR, neighbors, pairwise_mask, dR_expanded)
            x = x + v

        return x