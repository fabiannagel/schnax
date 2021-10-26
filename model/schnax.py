import haiku as hk
from jax_md.partition import NeighborList

import jax.numpy as jnp

from model.interaction.aggregate import Aggregate
from utils import shifted_softplus
from model.gaussian_smearing import GaussianSmearing
from model.interaction.interaction import Interaction


class Schnax(hk.Module):
    n_atom_basis = 128
    max_z = 100
    n_gaussians = 25

    n_filters = 128
    n_interactions = 1

    mean = 0.0
    stddev = 20.0

    # config_atomwise = {'n_in': 128, 'mean': 0.0, 'stddev': 20.0, 'n_layers': 2, 'n_neurons': None}

    def __init__(self, r_cutoff: float, per_atom: bool):
        super().__init__(name="SchNet")
        self.per_atom = per_atom

        self.embedding = hk.Embed(self.max_z, self.n_atom_basis, name="embeddings")  # TODO: Torch padding_idx missing in Haiku.
        self.distance_expansion = GaussianSmearing(0.0, r_cutoff, self.n_gaussians)

        self.interactions = hk.Sequential([
            Interaction(idx=i, n_atom_basis=self.n_atom_basis,
                        n_filters=self.n_filters, n_spatial_basis=self.n_gaussians,
                        r_cutoff=r_cutoff) for i in range(self.n_interactions)
        ])

        self.atomwise = hk.nets.MLP(output_sizes=[64, 1], activation=shifted_softplus, name="atomwise")
        self.aggregate = Aggregate(axis=0, mean_pooling=False)


    @staticmethod
    def standardize(yi: jnp.ndarray, mean: float, stddev: float):
        return yi * stddev + mean

    def __call__(self, dR: jnp.ndarray, Z: jnp.ndarray, neighbors: NeighborList, *args, **kwargs) -> jnp.ndarray:
        # TODO: Move hk.set_state() calls into layer modules. Use self.name as key.

        # get embedding for Z
        x = self.embedding(Z)
        hk.set_state("embedding", x)

        # expand interatomic distances
        dR_expanded = self.distance_expansion(dR)
        # hk.set_state("distance_expansion", dR_expanded)

        # compute interactions
        for i, interaction in enumerate(self.interactions.layers):
            v = interaction(x, dR, neighbors, dR_expanded)
            x = x + v

        # energy contributions
        yi = self.atomwise(x)
        yi = self.standardize(yi, self.mean, self.stddev)

        if self.per_atom:
            return yi

        y = self.aggregate(yi)
        return y