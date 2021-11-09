from unittest import TestCase

from schnax.utils.schnetkit import get_interaction_count


class InteractionTestCase(TestCase):

    def __init__(self, method_name: str, geometry_file: str, weights_file: str):
        super().__init__(method_name)
        self.geometry_file = geometry_file
        self.weights_file = weights_file
        self.n_interactions = get_interaction_count(weights_file)