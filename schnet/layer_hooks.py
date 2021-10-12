from typing import Callable, Dict, OrderedDict


def register_layer_hook(layer_outputs: Dict, layer, layer_name: str):
    hook = get_layer_hook(layer_outputs, layer_name)
    layer.register_forward_hook(hook)


def get_layer_hook(layer_outputs: Dict, layer_name: str) -> Callable:
    def hook(model, input, output):
        layer_outputs[layer_name] = output.detach()

    return hook


def register_representation_layer_hooks(layer_outputs: Dict, model: OrderedDict):
    # layer_member_names = list(dict(model.named_modules()).keys())[1:]
    register_layer_hook(layer_outputs, model.representation.embedding, "representation.embedding")
    register_layer_hook(layer_outputs, model.representation.distances, "representation.distances")
    register_layer_hook(layer_outputs, model.representation.distance_expansion, "representation.distance_expansion")

    for interaction_network_idx, interaction_network in enumerate(model.representation.interactions):
        base_name = "representation.interactions.{}.".format(interaction_network_idx)

        # cfconv layer
        register_layer_hook(layer_outputs, interaction_network.filter_network, base_name + "filter_network.1")
        register_layer_hook(layer_outputs, interaction_network.cutoff_network, base_name + "cutoff_network")
        register_layer_hook(layer_outputs, interaction_network.cfconv.in2f, base_name + "cfconv.in2f")
        register_layer_hook(layer_outputs, interaction_network.cfconv.f2out, base_name + "cfconv.f2out")
        register_layer_hook(layer_outputs, interaction_network.cfconv.agg, base_name + "cfconv.agg")

        # dense output layer
        register_layer_hook(layer_outputs, interaction_network.dense, base_name + "dense")


def register_output_layer_hooks(layer_outputs: Dict, model: OrderedDict):
    for output_module_idx, output_module in enumerate(model.output_modules):
        base_name = "output_modules.{}.".format(output_module_idx)

        # GetItem()
        register_layer_hook(layer_outputs, output_module.out_net[0], base_name + "0")

        # 2-layer MLP
        register_layer_hook(layer_outputs, output_module.out_net[1].out_net[0], base_name + "1.out_net.0")
        register_layer_hook(layer_outputs, output_module.out_net[1].out_net[1], base_name + "1.out_net.1")

        # ScaleShift()
        register_layer_hook(layer_outputs, output_module.standardize, base_name + "standardize")

        # Aggregate()
        register_layer_hook(layer_outputs, output_module.atom_pool, base_name + "atom_pool")