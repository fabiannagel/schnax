import numpy as np

from datasets import load_test_data
from mnist.jax.layerwise_predict import JaxClassifier
from mnist.pytorch.layerwise_predict import TorchClassifier

model_file="pytorch/pytorch_weights_mnist.torch"
test_images, test_labels = load_test_data()

torch_classifier = TorchClassifier(model_file)
jax_classifier = JaxClassifier(model_file)

for input in test_images:
    torch_prediction = torch_classifier.predict(input)
    jax_prediction = jax_classifier.predict(input)

    for torch_params, jax_params in zip(torch_classifier.layer_outputs.values(), jax_classifier.layer_outputs.values()):
        torch_params = np.array(torch_params)
        np.testing.assert_allclose(jax_params, torch_params, atol=1e-6)
