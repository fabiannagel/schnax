import haiku as hk
from jax import numpy as jnp


class Aggregate(hk.Module):
    """Pooling layer based on sum or average with optional masking.

        Args:
            axis (int): axis along which pooling is done.
            mean (bool, optional): if True, use average instead for sum pooling.
            keepdim (bool, optional): whether the output tensor has dim retained or not.

    """

    def __init__(self, axis: int, mean=False, keepdim=True):
        super().__init__(name="Aggregate")
        self.axis = axis
        self.average = mean
        self.keepdim = keepdim

    def __call__(self, input: jnp.ndarray, mask=None):
        # mask input
        if mask is not None:
            input = input * mask[..., None]

        # compute sum of input along axis
        y = jnp.sum(input, self.axis)

        # compute average of input along axis
        if self.average:

            # get the number of items along axis
            if mask is not None:
                N = jnp.sum(mask, self.axis, keepdim=self.keepdim)
                N = jnp.max(N, other=jnp.ones_like(N))

            else:
                # N = input.size(self.axis)
                N = input.shape(self.axis)

            y = y / N

        return y