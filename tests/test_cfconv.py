from unittest import TestCase

import numpy as np
import torch
from jax_md.partition import NeighborList
import jax.numpy as jnp

from schnax.model.interaction.cfconv import CFConv

import tests.test_utils.initialize as init
import tests.test_utils.activation as activation
from tests.interaction_test_case import InteractionTestCase


class CFConvTest(InteractionTestCase):
    r_cutoff = 5.0

    def __init__(self, method_name: str):
        super().__init__(method_name, geometry_file="tests/assets/zro2_n_96.in",
                         weights_file="tests/assets/model_n1.torch")

    def setUp(self):
        _, self.schnet_activations, __ = init.initialize_and_predict_schnet(
            self.geometry_file, self.weights_file, self.r_cutoff, sort_nl_indices=True
        )

        self.schnax_activations, self.schnax_neighbors = self.initialize_schnax()

    def initialize_schnax(self):
        R, Z, box, neighbors, displacement_fn = init.initialize_schnax(
            self.geometry_file, self.r_cutoff, sort_nl_indices=True
        )

        energy, energies, forces, schnax_activations = init.predict_schnax(
            R, Z, box, displacement_fn, neighbors, self.r_cutoff, self.weights_file, return_actiations=True
        )

        return schnax_activations, neighbors

    def test_filter_network(self):
        for i in range(self.n_interactions):
            schnet_interaction, schnax_interaction = activation.get_cfconv_filters(
                self.schnet_activations, self.schnax_activations, interaction_block_idx=i
            )

            relevant_schnax_interactions = schnax_interaction[:, 0:48, :]
            np.testing.assert_allclose(
                schnet_interaction, relevant_schnax_interactions, rtol=1e-6, atol=3 * 1e-6
            )

    def test_cutoff_network(self):
        for i in range(self.n_interactions):
            schnet_cutoff, schnax_cutoff = activation.get_cutoff_network(
                self.schnet_activations, self.schnax_activations, interaction_block_idx=i
            )
            self.assertEqual((96, 48), schnet_cutoff.shape)
            self.assertEqual((96, 60), schnax_cutoff.shape)

            relevant_schnax_cutoff = schnax_cutoff[:, 0:48]
            np.testing.assert_allclose(
                schnet_cutoff, relevant_schnax_cutoff, rtol=1e-6, atol=1e-6
            )

    def test_in2f(self):
        for i in range(self.n_interactions):
            schnet_in2f, schnax_in2f = activation.get_in2f(
                self.schnet_activations, self.schnax_activations, interaction_block_idx=i
            )

            np.testing.assert_allclose(schnet_in2f, schnax_in2f, rtol=1e-6, atol=3 * 1e-6)

    def test_reshaping_and_elementwise_product_equality(self):
        """Assert equality of reshaping logic and element-wise product (after in2f, before aggregation).
        As this operation is sensitive to the NL's padding strategy, we use schnax's neighbor list as input for both.

        To make it work with SchNetPack, we have to:
            (1) add a dimension for batches
            (2) change the padding strategy to something that works with how PyTorch indexes tensors
        """

        # TODO: Maybe this test could be improved.
        # Instead of copy&pasting SchNet reshaping logic from schnetpack/nn/cfconv.py, we could mock the entire class
        # and all unnecessary forward pass calls. That way, we'd be able to test reshaping & element-wise multiplication
        # while keeping the unit test more robust."""

        def schnet_reshape(schnet_in2f: jnp.ndarray, neighbors: NeighborList):
            def do_reshape(y: torch.Tensor, neighbors: torch.Tensor):
                """Reshape y for element-wise multiplication by W (filter block output).
                (n_batches, 96, 128) -> (n_batches, 96, 48, 128)

                Taken from schnetpack/nn/cfconv.py
                """
                import torch

                nbh_size = neighbors.size()

                # (n_batches, n_atoms, max_occupancy) -> (n_batches, n_atoms * max_occupancy, 1)
                nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
                # (n_batches, n_atoms * max_occupancy, 1) -> (n_batches, n_atoms * max_occupancy, n_filters)
                nbh = nbh.expand(-1, -1, y.size(2))
                # (n_batches, n_atoms, n_filters) -> (n_batches, n_atoms * max_occupancy, n_filters)
                y = torch.gather(y, 1, nbh)

                # (n_batches, n_atoms * max_occupancy, n_filters) -> (n_batches, n_atoms, max_occupancy, n_filters)
                y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)
                return y

            # convert to torch tensors and add a batch dimension for compatibility
            schnet_in2f = torch.tensor(schnet_in2f)[None, ...]

            self.assertEqual((96, 60), neighbors.idx.shape)
            neighbor_indices = neighbors.idx[:, 0:48]

            neighbors = torch.tensor(neighbor_indices, dtype=torch.int64)[None, ...]

            # replace padding by n_atoms through (n_atoms - 1) to prevent out-of-bound errors
            # background: JAX falls back to returning the last element (n_atoms - 1) when passed an index that is out-of-bounds.
            # to reproduce the same behavior in PyTorch, we have to explicitly provide the (n_atoms - 1) as padding.
            neighbors[neighbors == neighbors.shape[1]] = neighbors.shape[1] - 1

            return do_reshape(schnet_in2f, neighbors)

        for i in range(self.n_interactions):
            schnet_in2f, schnax_in2f = activation.get_in2f(
                self.schnet_activations, self.schnax_activations, interaction_block_idx=i
            )
            np.testing.assert_allclose(
                schnet_in2f, schnax_in2f, rtol=1e-6, atol=3 * 1e-6
            )

            schnet_in2f = schnet_reshape(schnet_in2f, self.schnax_neighbors)
            schnax_in2f = CFConv._reshape_y(schnax_in2f, self.schnax_neighbors)
            np.testing.assert_allclose(
                schnet_in2f[0], schnax_in2f[:, 0:48], rtol=1e-6, atol=3 * 1e-6
            )

            schnet_W, schnax_W = activation.get_cfconv_filters(
                self.schnet_activations, self.schnax_activations, interaction_block_idx=i
            )
            schnet_W = torch.tensor(schnet_W)[None, ...]  # add batches dimension

            schnet_y = schnet_in2f * schnet_W
            schnax_y = schnax_in2f * schnax_W

            # TODO: Test seems volatile. See above for improvements.
            np.testing.assert_allclose(
                schnet_y[0], schnax_y[:, 0:48], rtol=1e-5, atol=8 * 1e-6
            )

    def test_aggregate(self):
        for i in range(self.n_interactions):
            schnet_agg, schnax_agg = activation.get_aggregate(
                self.schnet_activations, self.schnax_activations, interaction_block_idx=i
            )
            np.testing.assert_allclose(
                schnet_agg, schnax_agg, rtol=5 * 1e-6, atol=5 * 1e-6
        )

    def test_f2out(self):
        for i in range(self.n_interactions):
            schnet_f2out, schnax_f2out = activation.get_f2out(
                self.schnet_activations, self.schnax_activations, interaction_block_idx=i
            )
            np.testing.assert_allclose(
                schnet_f2out, schnax_f2out, rtol=1e-6, atol=2 * 1e-6
            )
