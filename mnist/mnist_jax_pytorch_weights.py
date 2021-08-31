from typing import List
from jax import random
import jax.numpy as jnp
from jax.experimental import stax
import torch
import datasets


def get_layer_names(model) -> List[str]:
    replace = lambda k: k.replace(".weight", "").replace(".bias", "")
    unique_layer_names = dict.fromkeys(map(replace, model.keys()))
    return list(unique_layer_names)


def get_params(layer_name: str, model):
    weights = model[layer_name + ".weight"]
    if weights.device != "cpu":
        weights = weights.cpu()

    bias = model[layer_name + ".bias"]
    if bias.device != "cpu":
        bias = bias.cpu()

    return jnp.array(weights.numpy().T), jnp.array(bias.numpy())


def compute_forward_pass(layer_idx: int, input, layers, params):
    """
    Compute all forward passes up to a certain layer index for a list of stax layers and their respective parameters.
    """

    # for each layer, obtain the function that computes its forward pass
    forward_pass_fns = [layer[1] for layer in layers]

    for i in range(layer_idx+1):
        layer_params = params[i]
        input = forward_pass_fns[i](layer_params, input)

    return input

rng = random.PRNGKey(0)
_, __, test_images, test_labels = datasets.mnist()
model = torch.load("pytorch_weights_mnist.torch")

layers = [
    stax.Dense(1024), stax.Relu,
    stax.Dense(1024), stax.Relu,
    stax.Dense(10), stax.Softmax
]

# we don't care about init functions
_, predict = stax.serial(*layers)

# print(get_layer_names(model))
params = [
    get_params("fc1", model), (),
    get_params("fc2", model), (),
    get_params("fc3", model), ()
]

for idx in range(10):
    pred = compute_forward_pass(5, test_images[idx], layers, params)
    pred_class = jnp.argmax(pred)
    true_class = jnp.argmax(test_labels[idx])

    print("Predicted label: {}".format(pred_class))
    print("Ground truth: {}".format(true_class))
    print()
