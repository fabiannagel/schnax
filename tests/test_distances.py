from itertools import zip_longest
from unittest import TestCase

import numpy as np

import tests.test_utils.initialize as init
import utils

class DistancesTest(TestCase):
    r_cutoff = 5.0
    atol = 1e-6
    rtol = 1e-6

    def __init__(self, method_name: str):
        super().__init__(method_name)

    def setUp(self):
        self.schnet_R, self.schnet_nl, self.schnet_dR = self.initialize_schnet()
        self.schnax_R, self.schnax_nl, self.schnax_dR = self.initialize_schnax()

    def initialize_schnet(self):
        inputs, schnet_activations, preds = init.initialize_and_predict_schnet(sort_nl_indices=True)
        # skip batches for now
        R = inputs['_positions'][0].detach().numpy()
        nl = inputs['_neighbors'][0].detach().numpy()
        dR = schnet_activations['representation.distances'][0].numpy()
        return R, nl, dR

    def initialize_schnax(self):
        R, Z, box, neighbors, displacement_fn = init.initialize_schnax(r_cutoff=self.r_cutoff, sort_nl_indices=True)
        dR = utils.compute_distances(R, neighbors, displacement_fn)
        return R, neighbors.idx, dR

    def test_position_equality(self):
        self.assertEqual(self.schnet_R.shape, self.schnax_R.shape)
        np.testing.assert_allclose(self.schnet_R, self.schnax_R, atol=self.atol, rtol=self.rtol)

    def test_nl_shape_equality(self):
        self.assertEqual(self.schnet_nl.shape, self.schnax_nl.shape)

    def test_neighborhood_equality(self):
        """Asserts that every atom indices the same neighboring atoms in both neighbor list implementations."""

        # for the provided input data, we expect SchNetPack to default to a smaller neighborhood size than schnax.
        assert self.schnet_nl.shape[1] < self.schnax_nl.shape[1]

        # loop over neighborhoods
        for reference_atom_idx, (schnet_ngbhhood, schnax_ngbhhood) in enumerate(zip(self.schnet_nl, self.schnax_nl)):

            # loop over indices within a neighborhood. use fill values to account for different neighborhood sizes.
            for neighbor_atom_idx, (schnet_idx, schnax_idx) in enumerate(zip_longest(schnet_ngbhhood, schnax_ngbhhood, fillvalue=-1)):

                # fill values should not affect schnax_nl;
                # schnet_nl should have a smaller neighborhood size and require filling.
                assert schnax_idx != -1

                # mismatching indices within a neighborhood are acceptable if caused by
                # (1) different numerical values used for padding
                # (2) different neighborhood sizes, requiring fill values as we iterate over both neighborhoods.
                if not schnet_idx == schnax_idx:

                    # (1) In SchNetPack, a 0 is padding, if it does not occur at the neighborhoods 0-th position.
                    if schnet_idx == 0 and neighbor_atom_idx > 0:
                        # as long as schnax pads at the same position (using n_atoms), this is fine.
                        assert schnax_idx == self.schnax_nl.shape[0]

                    # (2) SchNetPack fill values should only pairwise match with padding in schnax
                    if schnet_idx != -1:
                        assert schnax_idx == self.schnax_nl.shape[0]


    def test_distances_metrics(self):
        self.assertEqual(np.min(self.schnet_dR), np.min(self.schnax_dR))
        self.assertEqual(np.max(self.schnet_dR), np.max(self.schnax_dR))
        np.testing.assert_allclose(np.sum(self.schnet_dR), np.sum(self.schnax_dR), rtol=1e-3, atol=self.atol)

    def test_distances_equality(self):
        assertion_failed = False

        for i, (dr_schnet, dr_schnax) in enumerate(zip(self.schnet_dR, self.schnax_dR)):

            try:
                np.testing.assert_allclose(dr_schnax, dr_schnet, rtol=self.rtol, atol=self.atol)
            except AssertionError:
                assertion_failed = True
                print("atom index = {}".format(i))
                pass

        if assertion_failed:
            self.fail()
