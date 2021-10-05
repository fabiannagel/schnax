import unittest

import numpy as np

import tests.test_utils
from tests import test_utils


class DistanceExpansionTest(unittest.TestCase):
    r_cutoff = 5.0
    atol = 1e-5
    rtol = 1e-5

    def setUp(self):
        _, schnet_activations, __ = test_utils.initialize_and_predict_schnet(sort_nl_indices=True)
        self.schnet_expansions = schnet_activations['representation.distance_expansion'][0].numpy()     # skip batches for now

        schnax_activations, _ = test_utils.initialize_and_predict_schnax(r_cutoff=self.r_cutoff, sort_nl_indices=True)
        self.schnax_expansions = test_utils.get_distance_expansion(schnax_activations)

    def test_distance_expansion_equality(self):
        # fails at distance expansions coming from neighborhoods around R[3], R[45] and R[71].
        # at least this is consistent with the problematic neighborhoods in test_distances.py
        assertion_failed = False

        for i, (expanded_dr_schnet, expanded_dr_schnax) in enumerate(zip(self.schnet_expansions, self.schnax_expansions)):

            for j, (schnet_neighbor, schnax_neighbor) in enumerate(zip(expanded_dr_schnet, expanded_dr_schnax)):

                try:
                    np.testing.assert_allclose(schnet_neighbor, schnax_neighbor, self.rtol, self.atol)

                except AssertionError:
                    assertion_failed = True

                    for k, (schnet_val, schnax_val) in enumerate(zip(schnet_neighbor, schnax_neighbor)):
                        try:
                            np.testing.assert_allclose(schnet_val, schnax_val, self.rtol, self.atol)
                        except AssertionError:
                            print(i, j, k)
                            pass

                    pass

        if assertion_failed:
            self.fail()

