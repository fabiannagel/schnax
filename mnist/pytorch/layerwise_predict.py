import torch
from mnist.datasets import load_test_data
from mnist.pytorch.train import MLP as torch_MLP
# from train import MLP as torch_MLP

# def get_torch_layer_names(model_state_dict) -> List[str]:
#     replace = lambda k: k.replace(".weight", "").replace(".bias", "")
#     unique_layer_names = dict.fromkeys(map(replace, model_state_dict.keys()))
#     return list(unique_layer_names)


def load_torch_model():
    state_dict = torch.load("pytorch_weights_mnist.torch")
    model = torch_MLP()
    model.load_state_dict(state_dict)
    model.eval()
    return model


def create_layerwise_hooks(torch_model):
    layer_outputs = {}
    def get_layer_output(name):
        def hook(model, input, output):
            layer_outputs[name] = output.detach()

        return hook

    layer_member_names = list(dict(torch_model.named_modules()).keys())[1:]

    # for every layer, register a hook to store data on forward pass (w/o activations!)
    for l in layer_member_names:
        torch_layer = getattr(torch_model, l)
        hook = get_layer_output(l)
        torch_layer.register_forward_hook(hook)

    return layer_outputs


test_images, test_labels = load_test_data(torch_tensor=True)
torch_model = load_torch_model()
layer_outputs = create_layerwise_hooks(torch_model)
output = torch_model(test_images[0])
