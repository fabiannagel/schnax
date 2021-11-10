import jax
import jax_md
import numpy as np
from ase.io import read
from jax import numpy as jnp
from jax_md.partition import NeighborList
from jax_md.space import DisplacementFn, Box
from schnetpack import AtomsConverter
from schnetpack.environment import AseEnvironmentProvider

from schnetkit import load

from schnax import energy, utils
from schnax.energy import schnet_neighbor_list
from schnax.utils.layer_hooks import (
    register_representation_layer_hooks,
    register_output_layer_hooks,
)
from schnax.utils.schnetkit import get_interaction_count
from .mock_environment_provider import MockEnvironmentProvider


def initialize_schnax(
    geometry_file="assets/geometry.in", r_cutoff=5.0, sort_nl_indices=False
):
    atoms = read(geometry_file)
    R, Z, box = utils.atoms_to_input(atoms)
    displacement_fn, shift_fn = jax_md.space.periodic_general(
        box, fractional_coordinates=False
    )

    neighbor_fn = jax_md.partition.neighbor_list(
        displacement_fn,
        box,
        r_cutoff,
        dr_threshold=0.0,  # as the effective cutoff = r_cutoff + dr_threshold
        mask_self=True,  # an atom is not a neighbor of itself
        fractional_coordinates=False,
    )

    neighbors = neighbor_fn(R)
    if sort_nl_indices:
        neighbors = sort_schnax_nl(neighbors)

    return R, Z, box, neighbors, displacement_fn


def sort_schnax_nl(neighbors: NeighborList) -> NeighborList:
    # constructing the NL with mask_self=True pads an *already existing* self-reference,
    # causing a padding index at position 0. sort in ascending order to move it to the end.
    new_indices = np.sort(neighbors.idx, axis=1)
    object.__setattr__(neighbors, 'idx', new_indices)
    return neighbors


def predict_schnax(
    R: jnp.ndarray,
    Z: jnp.ndarray,
    box: Box,
    displacement_fn: DisplacementFn,
    neighbors: NeighborList,
    r_cutoff: float,
    weights_file="assets/model_n1.torch",
    per_atom=False
):
    # n_interactions = get_interaction_count(weights_file)
    # init_fn, apply_fn = energy._get_model(displacement_fn, r_cutoff, n_interactions, per_atom)
    #
    # # get initial state and params from torch file
    # rng = jax.random.PRNGKey(0)
    # _, state = init_fn(rng, R, Z, neighbors)
    # params = utils.get_params(weights_file)
    #
    # # run forward pass and obtain intermediates
    # pred, state = apply_fn(params, state, R, Z, neighbors)
    # return state, pred

    neighbor_fn, init_fn, apply_fn = schnet_neighbor_list(displacement_fn, box, r_cutoff, dr_threshold=0.0,
                                                          n_interactions=get_interaction_count(weights_file),
                                                          per_atom=per_atom)

    rng = jax.random.PRNGKey(0)
    _, state = init_fn(rng, R, Z, neighbors)
    params = utils.get_params(weights_file)

    # run forward pass and obtain intermediates
    pred, state = apply_fn(params, state, R, Z, neighbors)
    return state, pred


def initialize_and_predict_schnax(
    geometry_file="assets/geometry.in",
    weights_file="assets/model_n1.torch",
    r_cutoff=5.0,
    sort_nl_indices=False,
    per_atom=False
):
    R, Z, box, neighbors, displacement_fn = initialize_schnax(
        geometry_file, r_cutoff, sort_nl_indices
    )
    return predict_schnax(
        R, Z, box, displacement_fn, neighbors, r_cutoff, weights_file, per_atom
    )


def get_schnet_inputs(
    geometry_file="assets/geometry.in", r_cutoff=5.0, mock_environment_provider=None
):
    atoms = read(geometry_file, format="aims")
    if not mock_environment_provider:
        converter = AtomsConverter(
            environment_provider=AseEnvironmentProvider(cutoff=r_cutoff), device="cpu"
        )
    else:
        converter = AtomsConverter(
            environment_provider=mock_environment_provider, device="cpu"
        )

    return converter(atoms)


def initialize_and_predict_schnet(
    geometry_file="assets/geometry.in",
    weights_file="assets/model_n1.torch",
    r_cutoff=5.0,
    sort_nl_indices=False,
):
    layer_outputs = {}

    mock_provider = None
    if sort_nl_indices:
        mock_provider = MockEnvironmentProvider(AseEnvironmentProvider(cutoff=r_cutoff))

    inputs = get_schnet_inputs(
        geometry_file, r_cutoff, mock_environment_provider=mock_provider
    )
    # inputs["_neighbor_mask"] = None

    model = load(weights_file)
    assert model.cutoff == r_cutoff
    model = model.model  # get raw schnetpack model without schnetkit wrapper
    register_representation_layer_hooks(layer_outputs, model)
    register_output_layer_hooks(layer_outputs, model)

    preds = model(inputs)
    return inputs, layer_outputs, preds
