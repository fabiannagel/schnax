from unittest import TestCase

import numpy as np
from ase.io import read
from schnetkit import Calculator

import tests.test_utils.initialize as init


class EndToEndTest(TestCase):
    geometry_file = "tests/assets/zro2_n_96.in"
    weights_file = "tests/assets/model_n1.torch"

    r_cutoff = 5.0

    def __init__(self, method_name: str):
        super().__init__(method_name)

    def setUp(self):
        self.schnet_preds = self._initalize_and_predict_schnet()
        self.schnax_preds = self._initialize_and_predict_schnax()

    def _initalize_and_predict_schnet(self):
        schnet = Calculator(self.weights_file, skin=0.0, energies=True, stress=False)
        atoms = read(self.geometry_file)
        preds = schnet.calculate(atoms)

        return {
            'energy': preds["energy"],
            'energies': preds["energies"],
            'forces': preds["forces"]
        }

    def _initialize_and_predict_schnax(self):
        energy, energies, forces = init.initialize_and_predict_schnax(
            self.geometry_file, self.weights_file, self.r_cutoff, sort_nl_indices=False, return_activations=False
        )

        return {
            'energy': energy,
            'energies': energies,
            'forces': forces
        }

    def test_energy_equality(self):
        np.testing.assert_allclose(
            self.schnet_preds['energy'], self.schnax_preds['energy'], rtol=1e-6, atol=4 * 1e-5
        )

    def test_energies_equality(self):
        np.testing.assert_allclose(
            self.schnet_preds['energies'], self.schnax_preds['energies'], rtol=1e-6, atol=1e-5
        )

    def test_forces_equality(self):
        np.testing.assert_allclose(
            self.schnet_preds['forces'], self.schnax_preds['forces'], rtol=1e-6, atol=3 * 1e-5
        )