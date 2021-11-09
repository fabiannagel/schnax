from typing import Dict

from schnetkit.engine import load_file
import regex as re


def get_interaction_count(file: str) -> int:
    _, state = load_file(file)
    layer_names = list(state.keys())
    regex = re.compile('representation\.interactions\.(?P<index>[0-9]+).*')
    matches = [regex.match(l) for l in layer_names]
    indices = [int(m.group('index')) for m in matches if m is not None]
    return max(indices) + 1


def get_params(file: str) -> Dict:
    """Read weights from an existing schnetkit torch model."""
    _, state = load_file(file)
    n_interactions = get_interaction_count(file)
    params = {}

    def get_param(key):
        return state[key].cpu().numpy()

    def set_params(layer_key: str, weight_key: str, bias_key=None, interaction_idx=None):
        if interaction_idx is not None:
            layer_key = layer_key.format(interaction_idx)
            weight_key = weight_key.format(interaction_idx)
            if bias_key:
                bias_key = bias_key.format(interaction_idx)

        params[layer_key] = {}
        params[layer_key]['w'] = get_param(weight_key).T
        if bias_key:
            params[layer_key]['b'] = get_param(bias_key)

    # embeddings layer (special case, no transpose)
    params['SchNet/~/embeddings'] = {
        'embeddings': get_param('representation.embedding.weight')
    }

    for i in range(n_interactions):
        # interaction block // cfconv block // filter network
        set_params(
            layer_key='SchNet/~/Interaction_{}/~/CFConv/~/FilterNetwork/~/linear_0',
            weight_key='representation.interactions.{}.filter_network.0.weight',
            bias_key='representation.interactions.{}.filter_network.0.bias',
            interaction_idx=i
        )

        set_params(
            layer_key='SchNet/~/Interaction_{}/~/CFConv/~/FilterNetwork/~/linear_1',
            weight_key='representation.interactions.{}.filter_network.1.weight',
            bias_key='representation.interactions.{}.filter_network.1.bias',
            interaction_idx=i
        )

        # interaction block // cfconv block // in2f
        set_params(
            layer_key='SchNet/~/Interaction_{}/~/CFConv/~/in2f',
            weight_key='representation.interactions.{}.cfconv.in2f.weight',
            interaction_idx=i
        )

        # interaction block // cfconv block // f2out
        set_params(
            layer_key='SchNet/~/Interaction_{}/~/CFConv/~/f2out',
            weight_key='representation.interactions.{}.cfconv.f2out.weight',
            bias_key='representation.interactions.{}.cfconv.f2out.bias',
            interaction_idx=i
        )

        # interaction block // output layer
        set_params(
            layer_key='SchNet/~/Interaction_{}/~/Output',
            weight_key='representation.interactions.{}.dense.weight',
            bias_key='representation.interactions.{}.dense.bias',
            interaction_idx=i
        )

    set_params(
        layer_key='SchNet/~/atomwise/~/linear_0',
        weight_key='output_modules.0.out_net.1.out_net.0.weight',
        bias_key='output_modules.0.out_net.1.out_net.0.bias',
    )
    set_params(
        layer_key='SchNet/~/atomwise/~/linear_1',
        weight_key='output_modules.0.out_net.1.out_net.1.weight',
        bias_key='output_modules.0.out_net.1.out_net.1.bias',
    )

    return params
