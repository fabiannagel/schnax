from typing import Tuple
from unittest import TestCase
import jax_md
import torch
from schnetpack.nn import AtomDistances

import utils
import utils as schnax_utils
from schnax import schnet_neighbor_list
import numpy as np


class DistancesComparison(TestCase):
    geometry_file = "../schnet/geometry.in"
    r_cutoff = 5.0
    dr_threshold = 1.0

    atol = 1e-6
    rtol = 1e-6

    def __init__(self, method_name: str):
        super().__init__(method_name)

    def setUp(self):
        self.schnetpack_R, self.schnetpack_nl, self.schnetpack_dR = self.get_schnetpack_input()
        self.schnax_R, self.schnax_nl, self.schnax_dR = self.get_schnax_nl()

    def get_schnetpack_input(self) -> Tuple[torch.tensor, torch.tensor]:
        from ase.io import read
        from schnet.convert import get_converter

        converter = get_converter(self.r_cutoff, "cpu")
        atoms = read(self.geometry_file, format="aims")
        inputs = converter(atoms)

        # compute distances
        dR = AtomDistances()(
            inputs['_positions'], inputs['_neighbors'], inputs['_cell'], inputs['_cell_offset'], neighbor_mask=inputs['_neighbor_mask']
        )

        return inputs['_positions'], inputs['_neighbors'], dR

    def get_schnax_nl(self):
        R, Z, box = schnax_utils.get_input(self.geometry_file, self.r_cutoff)
        displacement_fn, shift_fn = jax_md.space.periodic_general(box, fractional_coordinates=False)
        neighbor_fn, init_fn, apply_fn = schnet_neighbor_list(displacement_fn, box, self.r_cutoff, self.dr_threshold)

        # compute NL and distances
        neighbors = neighbor_fn(R)
        dR = utils.compute_nl_distances(displacement_fn, R, neighbors)

        return R, neighbors, dR

    def test_input_position_equality(self):
        schnetpack_R = self.schnetpack_R[0].numpy()
        self.assertEqual(schnetpack_R.shape, self.schnax_R.shape)
        np.testing.assert_allclose(schnetpack_R, self.schnax_R, atol=self.atol, rtol=self.rtol)

    def test_distances_equality(self):
        schnetpack_dR = self.schnetpack_dR[0].numpy()
        self.assertEqual(schnetpack_dR.shape, self.schnax_dR.shape)
        # np.testing.assert_allclose(schnetpack_dR, self.schnax_dR, atol=self.atol, rtol=self.rtol)
