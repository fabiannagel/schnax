from unittest import TestCase

import numpy as np
import torch
from jax_md.partition import NeighborList
import jax.numpy as jnp
import tests.test_utils.initialize as init
import tests.test_utils.activation as activation
from model.interaction.cfconv import CFConv


class CFConvTest(TestCase):
    """Asserts equal output of cfconv blocks as wholes.
    To bypass the fact that input neighbor lists are still not 100% equal, we temporarily use SchNetPack's representation (adapted to JAX-MD's conventions).
    That way, we can test an interaction block without having to deal with errors cascading down from the distance layer.

    TODO once test_distances.py and test_distance_expansion.py are passing: Use schnax's own NL.
    """
    geometry_file = "../schnet/geometry.in"
    weights_file = "../schnet/model_n1.torch"

    r_cutoff = 5.0
    atol = 1e-5
    rtol = 1e-5

    def __init__(self, method_name: str):
        super().__init__(method_name)

    def setUp(self):
        self.schnet_inputs, self.schnet_activations, __ = init.initialize_and_predict_schnet(geometry_file=self.geometry_file, weights_file=self.weights_file, r_cutoff=self.r_cutoff, sort_nl_indices=True)
        self.schnax_activations, self.schnax_neighbors = self.initialize_schnax()

    def initialize_schnax(self):
        """
        As long as we haven't achieved equal output from the NL layer, use SchNetPack's NL as input for schnax.
        To make the NL compatible to JAX-MD, we have to:

            (a) sort neighborhood indices in ascending order
            (b) replace SchNetPack's zero padding with JAX-MD's total atom count padding.
        """

        schnet_inputs, __, ___ = init.initialize_and_predict_schnet(geometry_file=self.geometry_file, weights_file=self.weights_file, r_cutoff=self.r_cutoff, sort_nl_indices=True)
        nl = schnet_inputs['_neighbors'][0].cpu().numpy()   # get rid of first dimension, no batches yet.

        mask = nl == 0          # zero valued indices are padding, ...
        mask[:, 0] = False      # unless they occur within the first column (as the NL is sorted).
        nl[mask] = nl.shape[0]  # use total atom count as padding.

        # initialize schnax and override NL
        R, Z, box, neighbors, displacement_fn = init.initialize_schnax(geometry_file=self.geometry_file, r_cutoff=self.r_cutoff)
        object.__setattr__(neighbors, 'idx', nl)

        layer_outputs, pred = init.predict_schnax(R, Z, displacement_fn, neighbors, self.r_cutoff, weights_file=self.weights_file)
        return layer_outputs, neighbors

    def test_distance_expansion(self):
        """Just a quick sanity check to see if we get approx. equal distance expansions when manually overriding schnax's NL."""
        # TODO once test_distances.py and test_distance_expansion.py are passing: Remove.
        schnet_edR, schnax_edR = activation.get_distance_expansion(self.schnet_activations, self.schnax_activations)
        np.testing.assert_allclose(schnet_edR, schnax_edR, self.rtol, self.atol)

    def test_filter_network(self):
        schnet_interaction, schnax_interaction = activation.get_cfconv_filters(self.schnet_activations, self.schnax_activations, interaction_block_idx=0)
        np.testing.assert_allclose(schnet_interaction, schnax_interaction, self.rtol, self.atol)

    def test_cutoff_network(self):
        schnet_cutoff, schnax_cutoff = activation.get_cutoff_network(self.schnet_activations, self.schnax_activations, interaction_block_idx=0)
        np.testing.assert_equal(schnet_cutoff, schnax_cutoff)

    def test_in2f(self):
        schnet_in2f, schnax_in2f = activation.get_in2f(self.schnet_activations, self.schnax_activations, interaction_block_idx=0)
        np.testing.assert_allclose(schnet_in2f, schnax_in2f, self.rtol, self.atol)

    def test_reshaping_and_elementwise_product_equality(self):
        """Assert equality of reshaping logic and element-wise product (after in2f, before aggregation).
        As this operation is sensitive to the NL's padding strategy, we use schnax's neighbor list as input for both.

        To make it work with SchNetPack, we have to:
            (1) add a dimension for batches
            (2) change the padding strategy to something that works with how PyTorch indexes tensors
        """

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
            neighbors = torch.tensor(neighbors.idx)[None, ...]

            # replace padding by n_atoms through (n_atoms - 1) to prevent out-of-bound errors
            # background: JAX falls back to returning the last element (n_atoms - 1) when passed an index that is out-of-bounds.
            # to reproduce the same behavior in PyTorch, we have to explicitly provide the (n_atoms - 1) as padding.
            neighbors[neighbors == neighbors.shape[1]] = neighbors.shape[1] - 1

            return do_reshape(schnet_in2f, neighbors)

        schnet_in2f, schnax_in2f = activation.get_in2f(self.schnet_activations, self.schnax_activations, interaction_block_idx=0)
        np.testing.assert_allclose(schnet_in2f, schnax_in2f, self.rtol, self.atol)

        schnet_in2f = schnet_reshape(schnet_in2f, self.schnax_neighbors)
        schnax_in2f = CFConv._reshape_y(schnax_in2f, self.schnax_neighbors)
        np.testing.assert_allclose(schnet_in2f[0], schnax_in2f, self.rtol, self.atol)

        schnet_W, schnax_W = activation.get_cfconv_filters(self.schnet_activations, self.schnax_activations, interaction_block_idx=0)
        schnet_W = torch.tensor(schnet_W)[None, ...]    # add batches dimension

        schnet_y = schnet_in2f * schnet_W
        schnax_y = schnax_in2f * schnax_W
        np.testing.assert_allclose(schnet_y[0], schnax_y, self.rtol, self.atol)

    def test_aggregate(self):
        schnet_agg, schnax_agg = activation.get_aggregate(self.schnet_activations, self.schnax_activations, interaction_block_idx=0)
        np.testing.assert_allclose(schnet_agg, schnax_agg, rtol=self.rtol, atol=5 * 1e-5)

    def test_f2out(self):
        # keep in mind: called after aggregation layer, thus also affected by pairwise mask.
        self.skipTest("not implemented yet")
