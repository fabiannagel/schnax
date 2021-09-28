from typing import Tuple
from unittest import TestCase
import jax_md
import torch
import numpy as np

from schnax import Schnax
from tests import test_utils


class EmbeddingsTest(TestCase):
    """Asserts output equality of both embedding layers, i.e. equal shape and approximate and exact equality of output tensors."""

    atol = 1e-6
    rtol = 1e-6

    def __init__(self, method_name: str):
        super().__init__(method_name)

    def setUp(self):
        _, schnet_activations, __ = test_utils.initialize_and_predict_schnet()
        self.schnet_embeddings = schnet_activations['representation.embedding'][0].numpy()  # skip batches for now

        _, schnax_activations = test_utils.initialize_and_predict_schnax()
        self.schnax_embeddings = schnax_activations['embedding']

    def test_embeddings_shape_equality(self):
        self.assertEqual(self.schnet_embeddings.shape, (96, 128))
        self.assertEqual(self.schnax_embeddings.shape, (96, 128))

    def test_embeddings_approx_equality(self):
        """Simply added as a safe guard to notice numerical instabilities (e.g. when enabling double precision in JAX)."""
        np.testing.assert_allclose(self.schnet_embeddings, self.schnax_embeddings, self.rtol, self.atol)

    def test_embeddings_exact_equality(self):
        """As the embedding layer simply performs a dictionary lookup from the same trained representations, the return values should be exactly equal."""
        np.testing.assert_equal(self.schnet_embeddings, self.schnax_embeddings)


