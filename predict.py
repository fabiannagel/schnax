import jax
import jax_md

import utils
from energy import schnet_neighbor_list


def predict(geometry_file: str):
    r_cutoff = 5.0
    dr_threshold = 0.0
    atoms, R, Z, box = utils.get_input(geometry_file, get_atoms=True)

    displacement_fn, shift_fn = jax_md.space.periodic_general(box, fractional_coordinates=False)
    neighbor_fn, init_fn, apply_fn = schnet_neighbor_list(displacement_fn, box, r_cutoff, dr_threshold, per_atom=True)
    # apply_fn = jax.jit(apply_fn)

    # compute neighbor list
    neighbors = neighbor_fn(R)

    # obtain PRNG key
    rng = jax.random.PRNGKey(0)

    # initialize model with a single example position and charge
    # we won't need these params as we will load the PyTorch model instead.
    # init_params = init_fn(rng, R, Z, neighbors)
    init_params, state = init_fn(rng, R, Z, neighbors)

    params = utils.get_params("schnet/model_n1.torch")
    pred, state = apply_fn(params, state, R, Z, neighbors)

    return pred, state


if __name__ == '__main__':
    energies, state = predict('schnet/geometry.in')

    print("e_pot = {} eV".format(energies))