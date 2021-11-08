from typing import Dict

from schnetkit.engine import load_file


def get_params(file: str) -> Dict:
    """Read weights from an existing schnetkit torch model."""
    _, state = load_file(file)

    def get_param(key):
        return state[key].cpu().numpy()

    params = {}

    def set_params(layer_key: str, weight_key: str, bias_key=None):
        params[layer_key] = {}
        params[layer_key]['w'] = get_param(weight_key).T
        if bias_key:
            params[layer_key]['b'] = get_param(bias_key)

    # embeddings layer (special case, no transpose)
    params['SchNet/~/embeddings'] = {
        'embeddings': get_param('representation.embedding.weight')
    }

    # interaction block // cfconv block // filter network
    set_params(
        layer_key='SchNet/~/Interaction_0/~/CFConv/~/FilterNetwork/~/linear_0',
        weight_key='representation.interactions.0.filter_network.0.weight',
        bias_key='representation.interactions.0.filter_network.0.bias',
    )
    set_params(
        layer_key='SchNet/~/Interaction_0/~/CFConv/~/FilterNetwork/~/linear_1',
        weight_key='representation.interactions.0.filter_network.1.weight',
        bias_key='representation.interactions.0.filter_network.1.bias',
    )

    # interaction block // cfconv block // in2f
    set_params(
        layer_key='SchNet/~/Interaction_0/~/CFConv/~/in2f',
        weight_key='representation.interactions.0.cfconv.in2f.weight',
    )

    # interaction block // cfconv block // f2out
    set_params(
        layer_key='SchNet/~/Interaction_0/~/CFConv/~/f2out',
        weight_key='representation.interactions.0.cfconv.f2out.weight',
        bias_key='representation.interactions.0.cfconv.f2out.bias',
    )

    # interaction block // output layer
    set_params(
        layer_key='SchNet/~/Interaction_0/~/Output',
        weight_key='representation.interactions.0.dense.weight',
        bias_key='representation.interactions.0.dense.bias',
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
