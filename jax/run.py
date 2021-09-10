from typing import OrderedDict, List

import torch
from ase.io import read
from schnetpack import Properties

from schnet.convert import get_converter


class Schnax:
    def __init__(self, torch_model_file: str, layer_wise_output_file: str):
        self.torch_model = self._load_torch_model(torch_model_file)
        self.params = self._load_params()
        self.layers = self._initialize_layers()

    def _load_torch_model(self, torch_model_file: str) -> OrderedDict:
        return torch.load(torch_model_file)

    def _load_params(self) -> List:
        return [
            # self.get_params("fc1", self.torch_model_state_dict), (),
            # self.get_params("fc2", self.torch_model_state_dict), (),
            # self.get_params("fc3", self.torch_model_state_dict), ()
        ]

    def _initialize_layers(self) -> List:
        # embedding layer
        

        pass

    def forward(self, inputs):
        atomic_numbers = inputs[Properties.Z]
        positions = inputs[Properties.R]
        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]
        atom_mask = inputs[Properties.atom_mask]




    def predict(self, geometry_file="geometry.in"):
        cutoff = 5.0  # note: cutoff is also defined in model config!
        device = "cpu"
        converter = get_converter(cutoff, device)
        atoms = read(geometry_file, format="aims")
        inputs = converter(atoms)

        # TODO: Run inference

