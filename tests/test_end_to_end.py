from unittest import TestCase

import numpy as np

import tests.test_utils.initialize as init


class EndToEndTest(TestCase):
    geometry_file = "../schnet/geometry.in"
    weights_file = "../schnet/model_n1.torch"

    r_cutoff = 5.0
    rtol = 1e-6
    atol = 1e-6

    def __init__(self, method_name: str):
        super().__init__(method_name)

    def setUp(self):
        inputs, layer_outputs, schnet_preds = init.initialize_and_predict_schnet(self.geometry_file, self.weights_file, self.r_cutoff)
        self.schnet_energy = schnet_preds['energy'][0].detach().numpy()
        self.schnet_energies = schnet_preds['energies'][0].detach().numpy()
        self.schnet_forces = schnet_preds['forces'][0].detach().numpy()

    def test_energy_equality(self):
        state, schnax_energy = init.initialize_and_predict_schnax(self.geometry_file, self.weights_file, self.r_cutoff, per_atom=False)
        np.testing.assert_allclose(self.schnet_energy, schnax_energy, rtol=6 * self.rtol, atol=self.atol)

    def test_energies_equality(self):
        state, schnax_energies = init.initialize_and_predict_schnax(self.geometry_file, self.weights_file, self.r_cutoff, per_atom=True)
        np.testing.assert_allclose(self.schnet_energies, schnax_energies, rtol=2 * self.rtol, atol=self.atol)

    # TODO: Test forces