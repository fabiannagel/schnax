import numpy as np

from datasets import load_test_data
import misc.mnist.haiku.layerwise_predict as haiku_classifier
from misc.mnist.pytorch.layerwise_predict import TorchClassifier

# np.testing.assert_allcose() compares the difference between actual and desired to atol + rtol * abs(desired).
rtol = 1e-6
atol = 1e-6

torch_model_file= "pytorch_weights_mnist.torch"
test_images, test_labels = load_test_data()

torch_classifier = TorchClassifier(torch_model_file)
# stax_classifier = StaxClassifier(torch_model_file)
haiku_params = haiku_classifier.get_params("pytorch_weights_mnist.torch")

for x in test_images:
    torch_prediction = torch_classifier.predict(x)
    # stax_prediction = stax_classifier.predict(input)
    haiku_prediction, haiku_layer_outputs = haiku_classifier.predict(x, haiku_params)

    # compare intermediate outputs of all layers
    for torch_layer_output, haiku_layer_output in zip(torch_classifier.layer_outputs, haiku_layer_outputs):
        np.testing.assert_allclose(haiku_layer_output, torch_layer_output, rtol=rtol, atol=atol)

    # compare final prediction
    np.testing.assert_allclose(haiku_prediction, torch_prediction, rtol=rtol, atol=atol)