import pickle
from typing import OrderedDict, Callable

from ase.io import read

from convert import get_converter
from model import load_model


def register_layerwise_hooks(model: OrderedDict):

    def register_layer_hook(layer, layer_name):
        hook = get_layer_hook(layer_name)
        getattr(layer, "register_forward_hook", hook)

    def get_layer_hook(layer_name: str) -> Callable:
        def hook(model, input, output):
            layer_outputs[layer_name] = output.detach()
        return hook

    # TODO: Manually hook into relevant layers? Or hook into all layers and check later what is needed?
    layer_member_names = list(dict(model.named_modules()).keys())[1:]

    model.representation.embedding.register_forward_hook(get_layer_hook("representation.embedding"))
    model.representation.distances.register_forward_hook(get_layer_hook("representation.distances"))
    model.representation.distance_expansion.register_forward_hook(get_layer_hook("representation.distance_expansion"))

    for interaction_network_idx, interaction_network in enumerate(model.representation.interactions):
        base_name = "representation.interactions.{}.".format(interaction_network_idx)

        # cfconv layer
        register_layer_hook(interaction_network.filter_network, base_name + "filter_network.0")
        register_layer_hook(interaction_network.filter_network, base_name + "filter_network.1")
        register_layer_hook(interaction_network.cutoff_network, "representation.interactions.0.cutoff_network")
        register_layer_hook(interaction_network.cfconv.in2f, base_name + "cfconv.in2f")
        register_layer_hook(interaction_network.cfconv.f2out, base_name + "cfconv.f2out")

        # dense output layer
        register_layer_hook(interaction_network.dense, base_name + "dense")

    # TODO: atomwise output network
    # for output_modules in model.output_modules:
        # two-layer output MLP
        # output_modules.out_net[1].out_net[0]
        # output_modules.out_net[1].out_net[1]

        # output_modules.standardize
        # output_modules.atom_pool


layer_outputs = {}
cutoff = 5.0  # note: cutoff is also defined in model config!
device = "cpu"

model = load_model("model_n1.torch", cutoff, device)
register_layerwise_hooks(model)

converter = get_converter(cutoff, device)
atoms = read("geometry.in", format="aims")
inputs = converter(atoms)
outputs = model(inputs)
print(outputs["energy"])

# pickle layer outputs
handle = open("model_n1_layer_outputs.pickle", "wb")
pickle.dump(layer_outputs, handle)
print(layer_outputs.keys())