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
        # _, schnet_activations, __ = test_utils.initialize_and_predict_schnet()
        # self.schnet_embeddings = schnet_activations['representation.embedding'][0].numpy()  # skip batches for now

        schnax_activations, _ = test_utils.initialize_and_predict_schnax()
        # self.schnax_embeddings = schnax_activations['interaction']

    def test_foo(self):
        pass
