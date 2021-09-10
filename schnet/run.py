import pickle
from typing import OrderedDict, Callable

from ase.io import read

from convert import get_converter
from model import load_model


def register_layer_hook(layer, layer_name):
    hook = get_layer_hook(layer_name)
    layer.register_forward_hook(hook)


def get_layer_hook(layer_name: str) -> Callable:
    def hook(model, input, output):
        layer_outputs[layer_name] = output.detach()

    return hook


def register_representation_layer_hooks(model: OrderedDict):
    # layer_member_names = list(dict(model.named_modules()).keys())[1:]
    register_layer_hook(model.representation.embedding, "representation.embedding")
    register_layer_hook(model.representation.distances, "representation.distances")
    register_layer_hook(model.representation.distance_expansion, "representation.distance_expansion")

    for interaction_network_idx, interaction_network in enumerate(model.representation.interactions):
        base_name = "representation.interactions.{}.".format(interaction_network_idx)

        # cfconv layer
        register_layer_hook(interaction_network.filter_network, base_name + "filter_network.0")
        register_layer_hook(interaction_network.filter_network, base_name + "filter_network.1")
        register_layer_hook(interaction_network.cutoff_network, base_name + "cutoff_network")
        register_layer_hook(interaction_network.cfconv.in2f, base_name + "cfconv.in2f")
        register_layer_hook(interaction_network.cfconv.f2out, base_name + "cfconv.f2out")

        # dense output layer
        register_layer_hook(interaction_network.dense, base_name + "dense")


def register_output_layer_hooks(model: OrderedDict):
    for output_module_idx, output_module in enumerate(model.output_modules):
        base_name = "output_modules.{}.".format(output_module_idx)

        # GetItem()
        register_layer_hook(output_module.out_net[0], base_name + "0")

        # 2-layer MLP
        register_layer_hook(output_module.out_net[1].out_net[0], base_name + "1.out_net.0")
        register_layer_hook(output_module.out_net[1].out_net[1], base_name + "1.out_net.1")

        # ScaleShift()
        register_layer_hook(output_module.standardize, base_name + "standardize")

        # Aggregate()
        register_layer_hook(output_module.atom_pool, base_name + "atom_pool")


layer_outputs = {}
cutoff = 5.0  # note: cutoff is also defined in model config!
device = "cpu"

model = load_model("model_n1.torch", cutoff, device)
register_representation_layer_hooks(model)
register_output_layer_hooks(model)

converter = get_converter(cutoff, device)
atoms = read("geometry.in", format="aims")
inputs = converter(atoms)
outputs = model(inputs)
print(outputs["energy"])

# pickle layer outputs
handle = open("model_n1_layer_outputs.pickle", "wb")
pickle.dump(layer_outputs, handle)
print(layer_outputs.keys())
