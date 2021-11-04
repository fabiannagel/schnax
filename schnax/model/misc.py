import jax
import jax.numpy as jnp


def shifted_softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softplus(x) - jnp.log(2.0)
