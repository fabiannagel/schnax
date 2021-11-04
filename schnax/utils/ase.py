import numpy as np
import jax.numpy as jnp
from ase import Atoms


def atoms_to_input(atoms: Atoms):
    R = atoms.positions.astype(np.float32)
    Z = atoms.numbers.astype(np.int)
    box = np.array(atoms.cell.array, dtype=np.float32)
    return jnp.float32(R), jnp.int32(Z), jnp.float32(box)
