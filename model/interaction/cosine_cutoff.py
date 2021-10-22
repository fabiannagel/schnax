import haiku as hk
import jax.numpy as jnp


class CosineCutoff(hk.Module):

    def __init__(self, r_cutoff: float):
        super().__init__(name="CosineCutoff")
        self.r_cutoff = jnp.float32(r_cutoff)

    def __call__(self, dR: jnp.ndarray) -> jnp.ndarray:
        cutoffs = 0.5 * (jnp.cos(dR * jnp.pi / self.r_cutoff) + 1.0)
        cutoffs *= (dR < self.r_cutoff)
        hk.set_state(self.name, cutoffs)
        return cutoffs
