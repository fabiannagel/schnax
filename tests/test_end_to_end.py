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

        state, schnax_preds = init.initialize_and_predict_schnax(self.geometry_file, self.weights_file, self.r_cutoff)
        self.schnax_energy = schnax_preds[0]
        self.schnax_energies = schnax_preds[1]
        self.schnax_forces = None   # TODO: Compute & compare forces

    def test_energy_equality(self):
        np.testing.assert_allclose(self.schnet_energy, self.schnax_energy, rtol=self.rtol, atol=2 * self.atol)

    def test_energies_equality(self):
        np.testing.assert_allclose(self.schnet_energies, self.schnax_energies, rtol=2 * self.rtol, atol=self.atol)
