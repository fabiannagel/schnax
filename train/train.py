import numpy as np
import optax
from jax import vmap, grad, jit, random, lax
from jax_md import space
from schnax.energy import schnet_neighbor_list
from schnax.utils.train import build_dataset, make_batches

# mkdir -p train/data/iso17
# !wget http://quantum-machine.org/datasets/iso17.tar.gz -P train/data/iso17
# !tar xzvf iso17.tar.gz
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
