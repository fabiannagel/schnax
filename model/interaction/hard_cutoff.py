import haiku as hk
import jax.numpy as jnp


class HardCutoff(hk.Module):

    def __init__(self, r_cutoff: float):
        super().__init__(name="HardCutoff")
        self.r_cutoff = r_cutoff

    def __call__(self, dR: jnp.ndarray):
        mask = (dR <= self.r_cutoff)
        hk.set_state(self.name, mask)
        return mask