from typing import OrderedDict, Tuple, Sequence, Any, List

import haiku as hk
import jax
import jax.numpy as jnp
import torch
from haiku._src.data_structures import FlatMapping
from jax import jit


class HaikuClassifier(hk.Module):

    def __init__(self):
        super().__init__(name="MNIST_classifier")
        self.layer_outputs = {}
        # tm = torch.load(torch_model_file)

        self.mlp = hk.Sequential([
            hk.Linear(1024), jax.nn.relu,
            hk.Linear(1024), jax.nn.relu,
            hk.Linear(10), jax.nn.softmax
        ])

    def _is_activation_function(self, layer):
        return "haiku" not in str(type(layer))

    def __call__(self, input: jnp.array):
        for layer in self.mlp.layers:
            layer_output = layer(input)

            if not self._is_activation_function(layer):
                k = str(layer.module_name)
                self.layer_outputs[k] = layer_output

            input = layer_output

        hk.set_state("layer_outputs", self.layer_outputs)
        return input
        # return self.mlp(x)


def forward(x: jnp.ndarray) -> jnp.array:
    classifier = HaikuClassifier()
    return classifier(x)


def get_layer_outputs(state: FlatMapping) -> List[jnp.ndarray]:
    layer_outputs = state['MNIST_classifier']['layer_outputs'].values()
    return list(layer_outputs)

@jit
def predict(x: jnp.ndarray, params) -> Tuple[jnp.ndarray, jnp.ndarray]:
    rng = jax.random.PRNGKey(0)
    net = hk.without_apply_rng(hk.transform_with_state(forward))

    # TODO: Don't obtain parameters via init functions. Obtain + convert them once in the file's global scope and simply reference them here.
    _, state = net.init(rng, x)
    pred, state = net.apply(params, state, x)
    return pred, get_layer_outputs(state)


def get_params(torch_model_file: str):
    torch_model = torch.load(torch_model_file)

    net = hk.without_apply_rng(hk.transform_with_state(forward))
    rng = jax.random.PRNGKey(0)
    x = jnp.ones(shape=(784,))
    params, state = net.init(rng, x)

    params = hk.data_structures.to_mutable_dict(params)

    params['MNIST_classifier/~/linear']['w'] = torch_model['fc1.weight'].cpu().numpy().T
    params['MNIST_classifier/~/linear']['b'] = torch_model['fc1.bias'].cpu().numpy()

    params['MNIST_classifier/~/linear_1']['w'] = torch_model['fc2.weight'].cpu().numpy().T
    params['MNIST_classifier/~/linear_1']['b'] = torch_model['fc2.bias'].cpu().numpy()

    params['MNIST_classifier/~/linear_2']['w'] = torch_model['fc3.weight'].cpu().numpy().T
    params['MNIST_classifier/~/linear_2']['b'] = torch_model['fc3.bias'].cpu().numpy()

    return params


# params = get_params("../pytorch_weights_mnist.torch")

# pred, layer_outputs = predict(jnp.ones(shape=(784,)), params)
# print(pred)
# print(layer_outputs)
