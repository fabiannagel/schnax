import haiku as hk
import jax.numpy as jnp


class GaussianSmearing(hk.Module):

    def __init__(self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False):
        super().__init__(name="GaussianSmearing")
        self.offset = jnp.linspace(start, stop, n_gaussians)
        self.widths = (self.offset[1] - self.offset[0]) * jnp.ones_like(self.offset)
        # TODO: trainable
        self.centered = centered

    def _smearing(self, distances: jnp.ndarray) -> jnp.ndarray:
        """Smear interatomic distance values using Gaussian functions."""

        if not self.centered:
            # compute width of Gaussian functions (using an overlap of 1 STDDEV)
            coeff = -0.5 / jnp.power(self.widths, 2)
            # Use advanced indexing to compute the individual components
            # diff = distances[:, :, :, None] - self.offset[None, None, None, :]
            diff = distances[:, :, None] - self.offset[None, None, :]   # skip batches for now
        else:
            # if Gaussian functions are centered, use offsets to compute widths
            coeff = -0.5 / jnp.power(self.offset, 2)
            # if Gaussian functions are centered, no offset is subtracted
            # diff = distances[:, :, :, None]
            diff = distances[:, :, None]    # skip batches for now

        # compute smear distance values
        gauss = jnp.exp(coeff * jnp.power(diff, 2))
        return gauss

    def __call__(self, distances: jnp.ndarray, *args, **kwargs):
        smearing = self._smearing(distances)
        hk.set_state(self.name, smearing)
        return smearing
