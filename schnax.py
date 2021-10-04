import haiku as hk
from jax_md.energy import DisplacementFn, Box
from jax_md.partition import NeighborList

import utils
import jax_md
import jax.numpy as jnp
import jax

from gaussian_smearing import GaussianSmearing
from interaction.interaction import Interaction


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
        hk.set_state("distance_expansion", dR_expanded)

        # compute interactions
        pairwise_mask = None  # TODO: Figure out what this is

        for i, interaction in enumerate(self.interactions.layers):
            v = interaction(x, dR, neighbors, pairwise_mask, dR_expanded)
            hk.set_state("interaction_{}".format(interaction.name), v)
            x = x + v

        return x


def _get_model(displacement_fn: DisplacementFn, r_cutoff: float):
    """Moved to dedicated method for better testing access."""

    @hk.without_apply_rng
    @hk.transform_with_state
    def model(R: jnp.ndarray, Z: jnp.int32, neighbors: jnp.ndarray):
        # dR = utils.compute_nl_distances(displacement_fn, R, neighbors)
        dR = utils.compute_distances_vectorized(R, neighbors, displacement_fn)
        net = Schnax(r_cutoff)
        return net(dR, Z, neighbors)

    return model




def schnet_neighbor_list(displacement_fn: DisplacementFn, box_size: Box, r_cutoff: float, dr_threshold: float):
    """Convenience wrapper around Schnax"""
    model = _get_model(displacement_fn, r_cutoff)

    neighbor_fn = jax_md.partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff,
        dr_threshold,
        capacity_multiplier=0.625,
        # capacity_multiplier=1.0,
        mask_self=False,
        fractional_coordinates=False)

    init_fn, apply_fn = model.init, model.apply
    return neighbor_fn, init_fn, apply_fn


def predict(geometry_file: str):
    r_cutoff = 5.0
    dr_threshold = 1.0
    R, Z, box = utils.get_input(geometry_file)

    displacement_fn, shift_fn = jax_md.space.periodic_general(box, fractional_coordinates=False)
    neighbor_fn, init_fn, apply_fn = schnet_neighbor_list(displacement_fn, box, r_cutoff, dr_threshold)
    apply_fn = jax.jit(apply_fn)

    # compute neighbor list
    neighbors = neighbor_fn(R)

    # obtain PRNG key
    rng = jax.random.PRNGKey(0)

    # initialize model with a single example position and charge
    # we won't need these params as we will load the PyTorch model instead.
    # init_params = init_fn(rng, R, Z, neighbors)
    init_params, state = init_fn(rng, R, Z, neighbors)
    print(list(init_params.keys()))

    params = utils.get_params("schnet/model_n1.torch")
    pred, state = apply_fn(params, state, R, Z, neighbors)

    return pred, state


if __name__ == '__main__':
    pred, state = predict("schnet/geometry.in")
    print(pred.shape)