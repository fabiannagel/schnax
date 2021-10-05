import jax
import jax_md
import numpy as np
from ase.io import read
from jax import numpy as jnp
from jax_md.partition import NeighborList
from jax_md.space import DisplacementFn
from schnetpack import AtomsConverter
from schnetpack.environment import AseEnvironmentProvider

import energy
import utils
from schnet.layer_hooks import register_representation_layer_hooks, register_output_layer_hooks
from schnet.model import load_model
from utils import get_input


def initialize_schnax(geometry_file="../schnet/geometry.in", r_cutoff=5.0, sort_nl_indices=False):
    R, Z, box = get_input(geometry_file)
    displacement_fn, shift_fn = jax_md.space.periodic_general(box, fractional_coordinates=False)

    neighbor_fn = jax_md.partition.neighbor_list(
        displacement_fn,
        box,
        r_cutoff,
        dr_threshold=0.0,  # as the effective cutoff = r_cutoff + dr_threshold
        capacity_multiplier=0.98,  # to match shapes with SchNet
        mask_self=True,  # an atom is not a neighbor of itself
        fractional_coordinates=False)

    neighbors = neighbor_fn(R)
    if sort_nl_indices:
        neighbors = sort_schnax_nl(neighbors)

    return R, Z, box, neighbors, displacement_fn


def predict_schnax(R: jnp.ndarray, Z: jnp.ndarray, displacement_fn: DisplacementFn, neighbors: NeighborList, r_cutoff: float, weights_file="../schnet/model_n1.torch"):
    init_fn, apply_fn = energy._get_model(displacement_fn, r_cutoff)

    # get initial state and params from torch file
    rng = jax.random.PRNGKey(0)
    _, state = init_fn(rng, R, Z, neighbors)
    params = utils.get_params(weights_file)

    # run forward pass and obtain intermediates
    pred, state = apply_fn(params, state, R, Z, neighbors)
    layer_outputs = state['SchNet']
    return layer_outputs, pred


def initialize_and_predict_schnax(geometry_file="../schnet/geometry.in", weights_file="../schnet/model_n1.torch", r_cutoff=5.0, sort_nl_indices=False):
    R, Z, box, neighbors, displacement_fn = initialize_schnax(geometry_file, r_cutoff, sort_nl_indices)
    return predict_schnax(R, Z, displacement_fn, neighbors, r_cutoff, weights_file)


def initialize_schnet(geometry_file="../schnet/geometry.in", r_cutoff=5.0, mock_environment_provider=None):
    atoms = read(geometry_file, format="aims")

    if not mock_environment_provider:
        converter = AtomsConverter(environment_provider=AseEnvironmentProvider(cutoff=r_cutoff), device="cpu")
    else:
        converter = AtomsConverter(environment_provider=mock_environment_provider, device="cpu")

    return converter(atoms)


def initialize_and_predict_schnet(geometry_file="../schnet/geometry.in", weights_file="../schnet/model_n1.torch", r_cutoff=5.0, sort_nl_indices=False):
    layer_outputs = {}

    mock_provider = None
    if sort_nl_indices:
        mock_provider = MockEnvironmentProvider(AseEnvironmentProvider(cutoff=r_cutoff))

    inputs = initialize_schnet(geometry_file, r_cutoff, mock_environment_provider=mock_provider)

    model = load_model(weights_file, r_cutoff, device="cpu")
    register_representation_layer_hooks(layer_outputs, model)
    register_output_layer_hooks(layer_outputs, model)

    preds = model(inputs)
    return inputs, layer_outputs, preds


class MockEnvironmentProvider:
    """Wraps around the default AseEnvironmentProvider to equalize NL conventions with JAX-MD.
    If we apply a consistent ordering to both the neighborhoods and offsets here, AtomsConverter() will implicitly apply it to all other inputs as well, making our life easier down the line."""

    def __init__(self, environment_provider: AseEnvironmentProvider):
        self.environment_provider = environment_provider

    def get_environment(self, atoms, **kwargs):
        neighborhood_idx, offset = self.environment_provider.get_environment(atoms, **kwargs)

        # replace -1 padding w/ atom count
        neighborhood_idx[neighborhood_idx == -1] = neighborhood_idx.shape[0]

        # that way, we can sort in ascending order and the padded indices stay "at the end"
        # as the same permutation has to be applied to offsets as well, we do this in two steps:
        # (1) sort and obtain indices of the new permutation. (2) apply the permutation to the neighborhoods.
        sorted_indices = np.argsort(neighborhood_idx, axis=1)
        neighborhood_idx = np.take_along_axis(neighborhood_idx, sorted_indices, axis=1)

        # reverse padding to -1 to stay compatible to the original AtomsConverter()
        # this gives us a SchNetPack-compatible NL with nice, ascending ordering and -1 padding (only!) at the end.
        # makes our life easier for comparing individual neighborhoods from both SchNet and schnax.
        neighborhood_idx[neighborhood_idx == neighborhood_idx.shape[0]] = -1

        # apply the same ordering to offsets.
        sorted_offset = np.empty_like(offset)
        for i, idx_row in enumerate(sorted_indices):
            for j, idx in enumerate(idx_row):

                matching_offset = offset[i][idx]
                sorted_offset[i][j] = matching_offset

        return neighborhood_idx, sorted_offset


def sort_schnax_nl(neighbors: NeighborList) -> NeighborList:
    # constructing the NL with mask_self=True pads an *already existing* self-reference,
    # causing a padding index at position 0. sort in ascending order to move it to the end.
    new_indices = np.sort(neighbors.idx, axis=1)
    object.__setattr__(neighbors, 'idx', new_indices)
    return neighbors


