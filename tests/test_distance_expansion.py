from unittest import TestCase

import numpy as np

import tests.test_utils.initialize as init
import tests.test_utils.activation as activation


class DistanceExpansionTest(TestCase):
    r_cutoff = 5.0
    atol = 1e-6
    rtol = 2 * 1e-5  # close to the edge - 1 * 1e-5 already fails.

    def setUp(self):
        _, schnet_activations, __ = init.initialize_and_predict_schnet(
            sort_nl_indices=True
        )

        energy, energies, forces, schnax_activations = init.initialize_and_predict_schnax(
            r_cutoff=self.r_cutoff, sort_nl_indices=True, return_activations=True
        )
        (
            self.schnet_expansions,
            self.schnax_expansions,
        ) = activation.get_distance_expansion(schnet_activations, schnax_activations)

    def test_distance_expansions(self):
        self.assertEqual((96, 48, 25), self.schnet_expansions.shape)
        self.assertEqual((96, 60, 25), self.schnax_expansions.shape)

        # SchNetPack's neighborhood size is 48, so we only compare up to that index.
        # anything past idx 48 is just padding anyways.
        relevant_schnax_expansions = self.schnax_expansions[:, 0:48, :]
        np.testing.assert_allclose(
            self.schnet_expansions,
            relevant_schnax_expansions,
            rtol=self.rtol,
            atol=self.atol,
        )
