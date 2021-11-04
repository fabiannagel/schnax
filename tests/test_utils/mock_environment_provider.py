import numpy as np
from schnetpack.environment import AseEnvironmentProvider


class MockEnvironmentProvider:
    """Wraps around the default AseEnvironmentProvider to equalize NL conventions with JAX-MD.
    If we apply a consistent ordering to both the neighborhoods and offsets here, AtomsConverter() will implicitly apply it to all other inputs as well, making our life easier down the line."""

    def __init__(self, environment_provider: AseEnvironmentProvider):
        self.environment_provider = environment_provider

    def get_environment(self, atoms, **kwargs):
        neighborhood_idx, offset = self.environment_provider.get_environment(
            atoms, **kwargs
        )

        # replace -1 padding w/ atom count
        neighborhood_idx[neighborhood_idx == -1] = neighborhood_idx.shape[0]

        # that way, we can sort in ascending order and the padded indices stay "at the end"
        # as the same permutation has to be applied to offsets as well, we do this in two steps:
        # (1) sort and obtain indices of the new permutation. (2) apply the permutation to the neighborhoods.
        sorted_indices = np.argsort(neighborhood_idx, axis=1)
        neighborhood_idx = np.take_along_axis(neighborhood_idx, sorted_indices, axis=1)

        # reverse padding to -1 to stay compatible to the original AtomsConverter()
        # this gives us a SchNetPack-compatible NL with nice, ascending ordering and -1 padding (only!) at the end.
        # makes our life easier for comparing individual neighborhoods from both SchNet and schnax.
        neighborhood_idx[neighborhood_idx == neighborhood_idx.shape[0]] = -1

        # apply the same ordering to offsets.
        sorted_offset = np.empty_like(offset)
        for i, idx_row in enumerate(sorted_indices):
            for j, idx in enumerate(idx_row):

                matching_offset = offset[i][idx]
                sorted_offset[i][j] = matching_offset

        return neighborhood_idx, sorted_offset
