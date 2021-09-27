import jax
import jax_md
from ase.io import read
from schnetpack import AtomsConverter
from schnetpack.environment import AseEnvironmentProvider

import schnax
import utils as schnax_utils
from schnet.layer_hooks import register_representation_layer_hooks, register_output_layer_hooks
from schnet.model import load_model


def initialize_schnax(geometry_file="../schnet/geometry.in", weights_file="../schnet/model_n1.torch", r_cutoff=5.0):
    R, Z, box = schnax_utils.get_input(geometry_file, r_cutoff)

    displacement_fn, shift_fn = jax_md.space.periodic_general(box, fractional_coordinates=False)

    neighbor_fn = jax_md.partition.neighbor_list(
        displacement_fn,
        box,
        r_cutoff,
        dr_threshold=0.0,  # important, as the applied cutoff = r_cutoff + dr_threshold
        capacity_multiplier=0.98,  # to match shapes with SchNet
        mask_self=True,
        fractional_coordinates=False)

    # compute NL and distances
    neighbors = neighbor_fn(R)

    # setup haiku model
    init_fn, apply_fn = schnax._get_model(displacement_fn, r_cutoff)

    rng = jax.random.PRNGKey(0)
    _, state = init_fn(rng, R, Z, neighbors)
    params = schnax_utils.get_params(weights_file)

    return params, state, apply_fn, (R, Z), neighbors


def initialize_and_predict_schnax(geometry_file="../schnet/geometry.in", weights_file="../schnet/model_n1.torch", r_cutoff=5.0):
    params, state, apply_fn, (R, Z), neighbors = initialize_schnax(geometry_file, weights_file, r_cutoff)
    pred, state = apply_fn(params, state, R, Z, neighbors)
    return pred, state['SchNet']


def initialize_and_predict_schnet(geometry_file="../schnet/geometry.in", weights_file="../schnet/model_n1.torch", r_cutoff=5.0):
    layer_outputs = {}
    device = "cpu"

    converter = AtomsConverter(environment_provider=AseEnvironmentProvider(cutoff=r_cutoff), device=device)
    atoms = read(geometry_file, format="aims")
    inputs = converter(atoms)

    model = load_model(weights_file, r_cutoff, device)
    register_representation_layer_hooks(layer_outputs, model)
    register_output_layer_hooks(layer_outputs, model)

    preds = model(inputs)
    return preds, layer_outputs
