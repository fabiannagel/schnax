from typing import Tuple

import numpy as np
import torch


def get_weights(layer_name: str, model) -> Tuple[np.array, np.array]:
    weights = model[layer_name + ".weight"]
    if weights.device != "cpu":
        weights = weights.cpu()

    bias = model[layer_name + ".bias"]
    if bias.device != "cpu":
        bias = bias.cpu()

    return weights.numpy(), bias.numpy()

# weights are stored as an ordered dictionary
model = torch.load("pytorch_weights_mnist.torch")

# every trainable layer has its own index for weights and biases respectively
# (hacky way to deduplicate keys while preserving their original order)
layer_names = list(dict.fromkeys(map(lambda k: k.replace(".weight", "").replace(".bias", ""), model.keys())))

# (1024, 784) and (1024,)
fc1_weights, fc1_bias = get_weights(layer_names[0], model)
print(fc1_weights.shape)
