import timeit
from functools import partial
from typing import Callable, Dict

import matplotlib.pyplot as plt
import jax
import jax_md
from matplotlib.ticker import MaxNLocator

import heat_flux
import utils
from energy import schnet_neighbor_list

"""
A simple benchmark to compare performance of heat flux implementations. 
"""

executions = 10
results = {}


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


def setup_forward_pass(jit: bool):
    apply_fn, params, state, R, Z, neighbors, atoms = initialize_schnax()
    forward_pass_fn = partial(apply_fn, params, state, R, Z, neighbors)

    if jit:
        forward_pass_fn = jax.jit(forward_pass_fn)

    # call block_until_ready() on result DeviceArray, not on haiku state.
    block_fn = lambda apply_fn: apply_fn()[0].block_until_ready()
    return partial(block_fn, forward_pass_fn)


def setup_heat_flux_fn(heat_flux_fn: Callable, jit: bool):
    """Returns a function to compute the heat flux without repeatedly initializing the underlying potential."""
    apply_fn, params, state, R, Z, neighbors, atoms = initialize_schnax()
    wrapped_heat_flux_fn = partial(heat_flux_fn, apply_fn, params, state, R, Z, neighbors, atoms)

    if jit:
        wrapped_heat_flux_fn = jax.jit(wrapped_heat_flux_fn)

    block_fn = lambda heat_flux_fn: heat_flux_fn().block_until_ready()
    return partial(block_fn, wrapped_heat_flux_fn)


def plot_results(results: Dict):
    x = list(range(executions))
    fig, ax = plt.subplots()

    for name, y in results.items():
        marker = 'o'
        if 'flux' not in name:
            marker = 'x'

        ax.plot(x, y, label=name, linestyle='dashed', marker=marker, markersize=6)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Consecutive executions")
    plt.ylabel("Computation time [s]")
    plt.yscale("log")
    plt.legend()
    plt.title("Heat flux computation time over consecutive executions\nPairwise distances in ASE")
    plt.show()
    plt.savefig("heat_flux.png")


def run_forward_pass(jit: bool):
    # baseline: forward pass only
    forward_pass_fn = setup_forward_pass(jit=jit)
    runtimes = timeit.repeat(forward_pass_fn, repeat=executions, number=1)
    print("Baseline (forward pass only), jit={}: {}".format(jit, runtimes))
    key = 'Forward pass only (jit={})'.format(jit)
    results[key] = runtimes


def run_heat_flux(jit: bool, heat_flux_fn: Callable, label: str):
    naive_heat_flux_fn = setup_heat_flux_fn(heat_flux_fn, jit=jit)
    runtimes = timeit.repeat(naive_heat_flux_fn, repeat=executions, number=1)

    print("{} (jit={}): {}".format(label, jit, runtimes))
    key = '{} (jit={})'.format(label, jit)
    results[key] = runtimes


run_forward_pass(jit=False)
run_forward_pass(jit=True)

run_heat_flux(jit=False, heat_flux_fn=heat_flux.compute_naive, label="Naive flux")
run_heat_flux(jit=True, heat_flux_fn=heat_flux.compute_naive, label="Naive flux")

run_heat_flux(jit=False, heat_flux_fn=heat_flux.compute_einsum, label="Einsum flux")
run_heat_flux(jit=True, heat_flux_fn=heat_flux.compute_einsum, label="Einsum flux")

plot_results(results)
