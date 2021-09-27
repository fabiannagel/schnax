import pickle

from ase.io import read

from convert import get_converter
from model import load_model
from schnet.layer_hooks import register_representation_layer_hooks, register_output_layer_hooks

layer_outputs = {}
cutoff = 5.0  # note: cutoff is also defined in model config!
device = "cpu"

model = load_model("model_n1.torch", cutoff, device)
register_representation_layer_hooks(layer_outputs, model)
register_output_layer_hooks(layer_outputs, model)

converter = get_converter(cutoff, device)
atoms = read("geometry.in", format="aims")
inputs = converter(atoms)
outputs = model(inputs)
print(outputs["energy"])

# pickle layer outputs
handle = open("model_n1_layer_outputs.pickle", "wb")
pickle.dump(layer_outputs, handle)
print(layer_outputs.keys())
