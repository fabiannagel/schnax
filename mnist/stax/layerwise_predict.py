from typing import OrderedDict, Tuple

import jax.numpy as jnp
import torch
from jax.experimental import stax


class StaxClassifier:

    def __init__(self, torch_model_file="pytorch_weights_mnist.torch"):
        self.torch_model_state_dict = self._load_torch_model(torch_model_file)
        self.layers = [
            stax.Dense(1024), stax.Relu,
            stax.Dense(1024), stax.Relu,
            stax.Dense(10), stax.Softmax
        ]

        # omit functions to randomly initialize layers
        _, self.layer_apply_fns = stax.serial(*self.layers)

        self.params = self.load_params()
        self.layer_outputs = {}

    def _load_torch_model(self, torch_model_file: str) -> OrderedDict:
        return torch.load(torch_model_file)

    def load_params(self):
        return [
            self.get_params("fc1", self.torch_model_state_dict), (),
            self.get_params("fc2", self.torch_model_state_dict), (),
            self.get_params("fc3", self.torch_model_state_dict), ()
        ]

    def get_params(self, layer_name: str, torch_model_state_dict) -> Tuple[jnp.array, jnp.array]:
        """
        For a given torch layer, retrieve weights and bias from the torch model.
        Transpose weights to match JAX/stax convention.
        """
        weights = torch_model_state_dict[layer_name + ".weight"]
        if weights.device != "cpu":
            weights = weights.cpu()

        bias = torch_model_state_dict[layer_name + ".bias"]
        if bias.device != "cpu":
            bias = bias.cpu()

        return jnp.array(weights.numpy().T), jnp.array(bias.numpy())

    # def get_torch_layer_names(self) -> List[str]:
    #     replace = lambda k: k.replace(".weight", "").replace(".bias", "")
    #     unique_layer_names = dict.fromkeys(map(replace, self.torch_model_state_dict.keys()))
    #     return list(unique_layer_names)

    def predict(self, input: jnp.array, store_activations=False):
        """
        Run a forward pass of the network and store layer-wise output features at self.layer_outputs.
        store_activations: Set to True if self.layer_outputs should include output of layer activation functions.
        """

        # layers are defined as tuples of init and feedforward functions
        # as we load existing PyTorch weights, we only need the forward pass functions
        pred_fns = [l[1] for l in self.layers]

        for i, (layer_pred_fn, layer_params) in enumerate(zip(pred_fns, self.params)):
            layer_output = layer_pred_fn(layer_params, input)

            # check if the current layer is an activation function
            # as PyTorch only returns layer outputs without applying activation functions, we might want to skip them for layer-wise comparisons to JAX

            if not "elementwise" in str(layer_pred_fn) or store_activations is True:
                self.layer_outputs[str(layer_pred_fn)] = layer_output

            input = layer_output

        return input
