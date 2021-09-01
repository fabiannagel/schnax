from typing import Dict
import torch
from mnist.pytorch.train import MLP as torch_MLP


class TorchClassifier:

    def __init__(self, model_file="pytorch_weights_mnist.torch"):
        self.model = self._load_torch_model(model_file)
        self.layer_outputs = self._create_layerwise_hooks()

    def _load_torch_model(self, model_file: str):
        state_dict = torch.load(model_file)
        model = torch_MLP()
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _create_layerwise_hooks(self) -> Dict:
        layer_outputs = {}
        def get_layer_output(name):
            def hook(model, input, output):
                layer_outputs[name] = output.detach()

            return hook

        layer_member_names = list(dict(self.model.named_modules()).keys())[1:]

        # for every layer, register a hook to store data on forward pass (w/o activations!)
        for l in layer_member_names:
            torch_layer = getattr(self.model, l)
            hook = get_layer_output(l)
            torch_layer.register_forward_hook(hook)

        return layer_outputs

    def predict(self, input: torch.tensor) -> torch.tensor:
        if type(input) != torch.Tensor:
            input = torch.tensor(input)
        pred = self.model(input)
        return pred
