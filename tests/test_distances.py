from typing import Tuple
from unittest import TestCase
import jax_md
import torch
from schnetpack import AtomsConverter
from schnetpack.environment import AseEnvironmentProvider
from schnetpack.nn import AtomDistances
from ase.io import read

import utils
import utils as schnax_utils
import numpy as np

from tests import test_utils


class MockEnvironmentProvider:
    """Wraps around the default AseEnvironmentProvider to equalize NL conventions with JAX-MD.
    If we apply a consistent ordering to both the neighborhoods and offsets here, AtomsConverter() will implicitly apply it to all other inputs as well, making our life easier down the line."""

    def __init__(self, environment_provider: AseEnvironmentProvider):
        self.environment_provider = environment_provider

    def get_environment(self, atoms, **kwargs):
        neighborhood_idx, offset = self.environment_provider.get_environment(atoms, **kwargs)

        # replace -1 padding w/ atom count
        neighborhood_idx[neighborhood_idx == -1] = neighborhood_idx.shape[0]

        # that way, we can sort in ascending order and the padded indices stay "at the end"
        # as the same permutation has to be applied to offsets as well, we do this in two steps:
        # (1) sort and obtain indices of the new permutation. (2) apply the permutation to the neighborhoods.
        sorted_indices = np.argsort(neighborhood_idx, axis=1)
        neighborhood_idx = np.take_along_axis(neighborhood_idx, sorted_indices, axis=1)

        # reverse padding to -1 to stay compatible to the original AtomsConverter()
        # this gives us a SchNetPack-compatible NL with nice, ascending ordering and -1 padding (only!) at the end.
        # makes our life easier for comparing individual neighborhoods from both SchNet and schnax.
        neighborhood_idx[neighborhood_idx == neighborhood_idx.shape[0]] = -1

        # apply the same ordering to offsets.
        sorted_offset = np.empty_like(offset)
        for i, idx_row in enumerate(sorted_indices):
            for j, idx in enumerate(idx_row):

                matching_offset = offset[i][idx]
                sorted_offset[i][j] = matching_offset

        return neighborhood_idx, sorted_offset


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
        mock_provider = MockEnvironmentProvider(AseEnvironmentProvider(cutoff=self.r_cutoff))
        inputs = test_utils.get_schnet_input(mock_environment_provider=mock_provider)

        # skip batches for now
        R = inputs['_positions'][0].numpy()
        nl = inputs['_neighbors'][0].numpy()

        # compute distances (layer designed for batches - use them).
        dR = AtomDistances()(
            inputs['_positions'], inputs['_neighbors'], inputs['_cell'], inputs['_cell_offset'],
            neighbor_mask=inputs['_neighbor_mask']
        )

        return R, nl, dR[0].numpy()

    def initialize_schnax(self):
        _, __, ___, (R, ____), neighbors, displacement_fn = test_utils.initialize_schnax()

        # constructing the NL with mask_self=True pads an *already existing* self-reference,
        # causing a padding index at position 0. sort in ascending order to move it to the end.
        sorted_indices = np.argsort(neighbors.idx, axis=1)
        nl = np.take_along_axis(neighbors.idx, sorted_indices, axis=1)

        # compute distances and apply the same reordering
        dR = utils.compute_distances_vectorized(R, neighbors, displacement_fn)
        dR = np.take_along_axis(dR, sorted_indices, axis=1)

        return np.array(R), nl, dR

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

    def test_distances_equality(self):

        for i, (dr_schnet, dr_schnax) in enumerate(zip(self.schnet_dR, self.schnax_dR)):

            try:
                np.testing.assert_allclose(dr_schnax, dr_schnet, rtol=self.rtol, atol=self.atol)
            except AssertionError:
                print("atom index = {}".format(i))
                self.fail()
