from unittest import TestCase

import numpy as np
from schnetkit import Calculator
from ase.io import read

import tests.test_utils.initialize as init


class EndToEndTest(TestCase):
    geometry_file = "assets/geometry.in"
    weights_file = "assets/model_n1.torch"

    r_cutoff = 5.0
    rtol = 1e-6
    atol = 2 * 1e-5

    def __init__(self, method_name: str):
        super().__init__(method_name)

    def setUp(self):
        calculator = Calculator(self.weights_file, skin=0.0, energies=True, stress=True)
        atoms = read(self.geometry_file)
        preds = calculator.calculate(atoms)
        self.schnet_energy = preds["energy"]
        self.schnet_energies = preds["energies"]
        self.schnet_forces = preds["forces"]

    def test_energy_equality(self):
        state, schnax_energy = init.initialize_and_predict_schnax(
            self.geometry_file, self.weights_file, self.r_cutoff, per_atom=False
        )
        np.testing.assert_allclose(
            self.schnet_energy, schnax_energy, rtol=6 * self.rtol, atol=self.atol
        )

    def test_energies_equality(self):
        state, schnax_energies = init.initialize_and_predict_schnax(
            self.geometry_file, self.weights_file, self.r_cutoff, per_atom=True
        )
        np.testing.assert_allclose(
            self.schnet_energies, schnax_energies, rtol=2 * self.rtol, atol=self.atol
        )

    # TODO: Test forces
