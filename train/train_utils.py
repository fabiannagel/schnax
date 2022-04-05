import numpy as np
from ase.db import connect
from jax import numpy as jnp


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