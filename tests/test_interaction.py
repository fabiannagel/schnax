from unittest import TestCase

import numpy as np

import tests.test_utils.initialize as init
import tests.test_utils.activation as activation
from tests.interaction_test_case import InteractionTestCase


class InteractionTest(InteractionTestCase):
    """Asserts equal output of interactions blocks as wholes as well as individual layers.
    To bypass the fact that input neighbor lists are still not 100% equal, we temporarily use SchNetPack's representation (adapted to JAX-MD's conventions).
    That way, we can test an interaction block without having to deal with errors cascading down from the distance layer.
    """
    r_cutoff = 5.0
    rtol = 1e-6
    atol = 1e-6

    def __init__(self, method_name: str):
        super().__init__(method_name, geometry_file="assets/geometry.in", weights_file="assets/model_n5.torch")

    def setUp(self):
        _, self.schnet_activations, _ = init.initialize_and_predict_schnet(
            geometry_file=self.geometry_file,
            weights_file=self.weights_file,
            r_cutoff=self.r_cutoff,
            sort_nl_indices=True,
        )
        self.schnax_activations, _ = init.initialize_and_predict_schnax(
            self.geometry_file, self.weights_file, self.r_cutoff, sort_nl_indices=True
        )

    def test_interaction_block(self):
        for i in range(self.n_interactions):
            schnet_interaction, schnax_interaction = activation.get_interaction_output(
                self.schnet_activations, self.schnax_activations, interaction_block_idx=i
            )
            np.testing.assert_allclose(
                schnet_interaction, schnax_interaction, rtol=self.rtol, atol=2 * self.atol
            )
