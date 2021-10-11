from unittest import TestCase

import numpy as np

import tests.test_utils.initialize as init
import tests.test_utils.activation as activation


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
        _, self.schnet_activations, __ = init.initialize_and_predict_schnet(geometry_file=self.geometry_file, weights_file=self.weights_file, r_cutoff=self.r_cutoff, sort_nl_indices=True)
        self.schnax_activations = self.initialize_schnax()

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
        return layer_outputs

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

    def test_aggregation(self):
        # keep in mind: this is where the pairwise mask is first used.
        self.skipTest()

    def test_f2out(self):
        # keep in mind: called after aggregation layer, thus also affected by pairwise mask.
        self.skipTest()
