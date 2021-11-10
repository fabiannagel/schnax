from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
from schnax import utils
from schnetkit import Calculator
from ase.io import read

import tests.test_utils.initialize as init
from schnax.utils.schnetkit import initialize_from_schnetkit_model


class EndToEndTest(TestCase):
    geometry_file = "assets/geometry.in"
    weights_file = "assets/model_n1.torch"

    r_cutoff = 5.0

    def __init__(self, method_name: str):
        super().__init__(method_name)

    def setUp(self):
        schnet = Calculator(self.weights_file, skin=0.0, energies=True, stress=True)
        atoms = read(self.geometry_file)
        preds = schnet.calculate(atoms)
        self.schnet_energy = preds["energy"]
        self.schnet_energies = preds["energies"]
        self.schnet_forces = preds["forces"]

    def test_energy_equality(self):
        atoms = read(self.geometry_file)
        R, Z, box = utils.atoms_to_input(atoms)

        params, neighbor_fn, init_fn, apply_fn = initialize_from_schnetkit_model(self.weights_file,
                                                                                 box=box,
                                                                                 dr_threshold=0.0,
                                                                                 per_atom=False)
        neighbors = neighbor_fn(R)

        rng = jax.random.PRNGKey(0)
        _, state = init_fn(rng, R, Z, neighbors)
        energy, state = apply_fn(params, state, R, Z, neighbors)

        np.testing.assert_allclose(
            self.schnet_energy, energy, rtol=6 * 1e-6, atol=1e-6
        )