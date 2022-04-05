import os.path

import numpy as np
import optax
import requests
from ase.db import connect
from jax import vmap, grad, jit, random, lax
import jax.numpy as jnp
from jax_md import space
from schnax.energy import schnet_neighbor_list


def build_dataset():

    def extract_properties():
        positions = []
        charges = []
        energies = []
        forces = []

        db = connect("train/data/iso17/reference.db")
        for row in db.select(limit=350):
            atoms = row.toatoms()

            positions += [atoms.get_positions()]
            charges += [atoms.get_atomic_numbers()]
            energies += [row['total_energy']]
            forces += [row.data['atomic_forces']]

        return jnp.float32(positions), jnp.int32(charges), jnp.float32(energies), jnp.float32(forces)

    def shuffle_dataset(positions, charges, energies, forces):
        dataset_size = positions.shape[0]
        lookup = np.arange(dataset_size)
        np.random.shuffle(lookup)
        permute = lambda a, lookup: jnp.take_along_axis(a, lookup, axis=0)
        return permute(positions, lookup[:, None, None]), permute(charges, lookup[:, None]), permute(energies, lookup), permute(forces, lookup[:, None, None])

    def split_dataset(positions, charges, energies, forces, train_size=0.8, val_size=0.1, test_size=0.1):
        assert train_size + val_size + test_size == 1.0
        dataset_size = positions.shape[0]
        idx_train_end = int(dataset_size * train_size)
        idx_val_start = int(dataset_size * train_size + dataset_size * val_size)

        train_positions, val_positions, test_positions = jnp.split(positions, [idx_train_end, idx_val_start])
        train_charges, val_charges, test_charges = jnp.split(charges, [idx_train_end, idx_val_start])
        train_energies, val_energies, test_energies = jnp.split(energies, [idx_train_end, idx_val_start])
        train_forces, val_forces, test_forces = jnp.split(forces, [idx_train_end, idx_val_start])

        train_set = train_positions, train_charges, train_energies, train_forces
        val_set = val_positions, val_charges, val_energies, val_forces
        test_set = test_positions, test_charges, test_energies, test_forces
        return train_set, val_set, test_set

    properties = extract_properties()
    properties = shuffle_dataset(*properties)
    train_set, val_set, test_set = split_dataset(*properties)
    # train_set = make_batches(*train_set, batch_size)
    return train_set, val_set, test_set


def make_batches(train_positions, train_charges, train_energies, train_forces, batch_size):
    batch_positions = []
    batch_charges = []
    batch_energies = []
    batch_forces = []

    dataset_size = train_positions.shape[0]
    lookup = jnp.arange(0, dataset_size)
    for i in range(0, dataset_size, batch_size):
        if i + batch_size > len(lookup):
            break

        idx = lookup[i:i + batch_size]

        batch_positions += [train_positions[idx]]
        batch_charges += [train_charges[idx]]
        batch_energies += [train_energies[idx]]
        batch_forces += [train_forces[idx]]

    return jnp.stack(batch_positions), jnp.stack(batch_charges), jnp.stack(batch_energies), jnp.stack(batch_forces)


train_set, val_set, test_set = build_dataset()
train_positions, train_charges, train_energies, train_forces = train_set
val_positions, val_charges, val_energies, val_forces = val_set
test_positions, test_charges, test_energies, test_forces = test_set

# TODO: What box size? Doesn't seem to be included in ISO17
box_size = 10.862
r_cutoff = 3.0
dr_threshold = 0.1

displacement_fn, shift_fn = space.periodic_general(box_size, fractional_coordinates=False)
neighbor_fn, init_fn, energy_fn = schnet_neighbor_list(displacement_fn, box_size, r_cutoff, dr_threshold)
neighbor = neighbor_fn.allocate(train_positions[0])


@jit
def train_energy_fn(params, R, Z):
    _neighbor = neighbor.update(R)
    return energy_fn(params, R, Z, _neighbor)


# Vectorize over states, not parameters
vectorized_energy_fn = vmap(train_energy_fn, (None, 0, 0))
grad_fn = grad(train_energy_fn, argnums=1)
force_fn = lambda params, R, Z, **kwargs: -grad_fn(params, R, Z)
vectorized_force_fn = vmap(force_fn, (None, 0, 0))

# Initialize random parameters
key = random.PRNGKey(0)
params = init_fn(key, train_positions[0], train_charges[0], neighbor=neighbor)

n_predictions = 500
example_positions = train_positions[:n_predictions]
example_charges = train_charges[:n_predictions]
example_energies = train_energies[:n_predictions]
example_forces = train_forces[:n_predictions]

predicted = vmap(train_energy_fn, (None, 0, 0))(params, example_positions, example_charges)

# seemingly no correlation from model priors - surprising?
# import matplotlib.pyplot as plt
# plt.plot(example_energies, predicted, 'o')
# plt.show()

@jit
def energy_loss(params, R, Z, energies):
    return np.mean((vectorized_energy_fn(params, R, Z) - energies) ** 2)

@jit
def force_loss(params, R, Z, forces):
    dforces = vectorized_force_fn(params, R, Z) - forces
    return np.mean(np.sum(dforces ** 2, axis=(1, 2)))

@jit
def loss(params, R, Z, energies, forces):
    return energy_loss(params, R, Z, energies) + force_loss(params, R, Z, forces)


opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))

@jit
def update_step(params, opt_state, R, Z, energies, forces):
    updates, opt_state = opt.update(grad(loss)(params, R, Z, energies, forces), opt_state)
    return optax.apply_updates(params, updates), opt_state

@jit
def update_epoch(params_and_opt_state, batches):
    def inner_update(params_and_opt_state, batch):
        params, opt_state = params_and_opt_state
        # b_xs, b_labels = batch
        # return update_step(params, opt_state, b_xs, b_labels), 0

        training_points, labels = batch
        positions, charges = training_points
        energies, forces = labels
        return update_step(params, opt_state, positions, charges, energies, forces), 0

    return lax.scan(inner_update, params_and_opt_state, batches)[0]


batch_positions, batch_charges, batch_energies, batch_forces = make_batches(train_positions, train_charges, train_energies, train_forces, batch_size=128)
train_epochs = 20
opt_state = opt.init(params)

train_energy_error = []
test_energy_error = []

for iteration in range(train_epochs):
    train_energy_error += [float(np.sqrt(energy_loss(params, batch_positions[0], batch_charges[0], batch_energies[0])))]
    # test_energy_error += [float(np.sqrt(energy_loss(params, , test_energies)))]
    # draw_training(params)

    params, opt_state = update_epoch((params, opt_state),
                                     ((batch_positions, batch_charges), (batch_energies, batch_forces)))

    # why shuffle here?
    # np.random.shuffle(lookup)
    # batch_positions, batch_charges, batch_energies, batch_forces = make_batches(lookup)

    print("Epoch {}/{}".format(iteration, train_epochs))
    print("Training error: {}".format(train_energy_error[-1]))
