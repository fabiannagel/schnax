import haiku as hk
import jax.numpy as jnp


class GaussianSmearing(hk.Module):

    def __init__(self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False):
        self.offset = jnp.linspace(start, stop, n_gaussians)
        self.widths = (self.offset[1] - self.offset[0]) * jnp.ones_like(self.offset)

        # TODO: trainable
        self.centered = centered

    def _smearing(self, distances):

        if not self.centered:
            coeff = -0.5 / jnp.power(self.widths, 2)
            diff = distances[:, :, :, None] - self.offset[None, None, None, :]
        else:
            coeff = -0.5 / jnp.power(self.offset, 2)
            diff = distances[:, :, :, None]

        gauss = jnp.exp(coeff * jnp.power(diff, 2))
        return gauss

    def __call__(self, distances: jnp.ndarray, *args, **kwargs):
        return self._smearing(distances)
