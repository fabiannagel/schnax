from unittest import TestCase

import numpy as np

import utils
from tests import test_utils


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
        inputs, schnet_activations, preds = test_utils.initialize_and_predict_schnet(sort_nl_indices=True)
        # skip batches for now
        R = inputs['_positions'][0].detach().numpy()
        nl = inputs['_neighbors'][0].detach().numpy()
        dR = schnet_activations['representation.distances'][0].numpy()
        return R, nl, dR

    def initialize_schnax(self):
        R, Z, box, neighbors, displacement_fn = test_utils.initialize_schnax(r_cutoff=self.r_cutoff, sort_nl_indices=True)
        dR = utils.compute_distances(R, neighbors, displacement_fn)
        return R, neighbors.idx, dR

    def test_position_equality(self):
        self.assertEqual(self.schnet_R.shape, self.schnax_R.shape)
        np.testing.assert_allclose(self.schnet_R, self.schnax_R, atol=self.atol, rtol=self.rtol)

    def test_nl_shape_equality(self):
        self.assertEqual(self.schnet_nl.shape, self.schnax_nl.shape)

    def test_neighborhood_equality(self):
        """Asserts that every atom indices the same neighboring atoms in both neighbor list implementations."""

        # we need to replace 0-padding with shape[0] to match JAX-MD's convention.
        # but at this point, we have both paddings with 0 as well as actual zero-valued indices!

        # as we sorted the neighbor list in ascending order in MockEnvironmentProvider, any padding zeros will be at a neighborhood's end.
        # if there is an actual index 0 within the neighbor list, it has to be at position 0.
        # thus, we should be safe if we replace all zeros beyond the 0-th position with shape[0] to match JAX-MD's NL convention.

        mask = self.schnet_nl == 0
        only_padding_neighborhood = np.all(self.schnet_nl == 0, axis=1)

        # as we sorted in ascendingly before, the first index cannot be padding. override mask to False.
        # rare exception: there is at least one "neighborhood" which entirely consists out of padding.
        # in that case, we'd have to selectively override the mask per-neighborhood.
        # however, with periodic materials, this is highly unlikely.
        if not np.any(only_padding_neighborhood):
            mask[:, 0] = False
        else:
            raise RuntimeError("Test not designed for sparse neighborhoods (see comments).")

        self.schnet_nl[mask] = self.schnet_nl.shape[0]

        for i, (neighbor_indices_schnet, neighbor_indices_schnax) in enumerate(zip(self.schnet_nl, self.schnax_nl)):
            try:
                if not np.all(neighbor_indices_schnet == neighbor_indices_schnax):
                    print("break")

                np.testing.assert_equal(neighbor_indices_schnet, neighbor_indices_schnax)

            except AssertionError:
                self.fail()

            # everything looks good apart from only 3 neighborhoods.
            # within these neighborhoods, only a single neighbor index is different.
            # schnet considers those a regular neighbor, whereas schnax applies padding.

            # considered a neighbor by SchNet, padded by JAX-MD
            # atom 3 -> 94
            # atom 45 -> 95
            # atom 71 -> 95

            # TODO: Check distances. are they close to cutoff range?
            # TODO: symmetrical issues?

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
