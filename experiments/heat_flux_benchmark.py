import timeit
from functools import partial
from typing import Callable

import jax
import jax_md

import heat_flux
import utils
from energy import schnet_neighbor_list

"""
A simple benchmark to compare performance of heat flux implementations. 
"""

executions = 2


def initialize_schnax():
    r_cutoff = 5.0
    dr_threshold = 0.0

    atoms, R, Z, box = utils.get_input('../schnet/geometry.in', get_atoms=True)
    displacement_fn, shift_fn = jax_md.space.periodic_general(box, fractional_coordinates=False)

    neighbor_fn, init_fn, apply_fn = schnet_neighbor_list(displacement_fn, box, r_cutoff, dr_threshold, per_atom=True)
    neighbors = neighbor_fn(R)

    rng = jax.random.PRNGKey(0)
    _, state = init_fn(rng, R, Z, neighbors)

    params = utils.get_params("../schnet/model_n1.torch")
    return apply_fn, params, state, R, Z, neighbors, atoms


def setup_heat_flux_fn(heat_flux_fn: Callable, jit: bool):
    """Returns a function to compute the heat flux without repeatedly initializing the underlying potential."""
    apply_fn, params, state, R, Z, neighbors, atoms = initialize_schnax()
    wrapped_heat_flux_fn = partial(heat_flux_fn, apply_fn, params, state, R, Z, neighbors, atoms)

    if jit:
        wrapped_heat_flux_fn = jax.jit(wrapped_heat_flux_fn)

    block_fn = lambda heat_flux_fn: heat_flux_fn().block_until_ready()
    return partial(block_fn, wrapped_heat_flux_fn)


def setup_forward_pass(jit: bool):
    apply_fn, params, state, R, Z, neighbors, atoms = initialize_schnax()
    forward_pass_fn = partial(apply_fn, params, state, R, Z, neighbors)

    if jit:
        forward_pass_fn =  jax.jit(forward_pass_fn)

    # call block_until_ready() on result DeviceArray, not on haiku state.
    block_fn = lambda apply_fn: apply_fn()[0].block_until_ready()
    return partial(block_fn, forward_pass_fn)


# naive heat flux, jit=False
naive_heat_flux_fn = setup_heat_flux_fn(heat_flux.compute_naive, jit=False)
runtimes = timeit.repeat(naive_heat_flux_fn, repeat=executions, number=1)
print("Naive heat flux (jit=False): {}".format(runtimes))
# jit=False [7.930954971932806, 0.39774937299080193]

# naive heat flux, jit=True
naive_heat_flux_fn = setup_heat_flux_fn(heat_flux.compute_naive, jit=True)
runtimes = timeit.repeat(naive_heat_flux_fn, repeat=executions, number=1)
print("Naive heat flux (jit=True): {}".format(runtimes))
# jit=True  [2.3618956330465153, 0.048938403953798115]

# baseline: forward pass only, jit=False
forward_pass_fn = setup_forward_pass(jit=False)
runtimes = timeit.repeat(forward_pass_fn, repeat=executions, number=1)
print("Baseline (forward pass only), jit=False: {}".format(runtimes))
# jit=False [0.06963449402246624, 0.03619032702408731]

forward_pass_fn = setup_forward_pass(jit=True)
runtimes = timeit.repeat(forward_pass_fn, repeat=executions, number=1)
print("Baseline (forward pass only), jit=True: {}".format(runtimes))
# jit=True  [0.4182406410109252, 0.004909868002869189]
