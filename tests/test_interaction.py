from typing import Tuple
from unittest import TestCase
import jax_md
import torch
import numpy as np

from schnax import Schnax
from tests import test_utils


class InteractionTest(TestCase):
    atol = 1e-6
    rtol = 1e-6

    def __init__(self, method_name: str):
        super().__init__(method_name)

    def setUp(self):
        _, schnet_activations, __ = test_utils.initialize_and_predict_schnet()
        self.schnet_interaction_0 = schnet_activations['representation.interactions.0.dense'][0].numpy()  # skip batches for now

        # TODO: test schnax w/ SchNetPack's NL (no differences there)

        # TODO: use neighborhood mask/pairwise mask for cfconv layer in schnax (currently not the case)

        # TODO: unit tests for individual layers of interaction block

        schnax_activations, _ = test_utils.initialize_and_predict_schnax()
        self.schnax_interaction_0 = schnax_activations['Interaction_0']

    def test_foo(self):
        np.testing.assert_allclose(self.schnet_interaction_0, self.schnax_interaction_0, self.rtol, self.atol)
