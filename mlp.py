from typing import Tuple

from jax_primitives import modelclass, Linear, Dynamic
import jax.numpy as jnp
import jax


def sinusoidal_emb(x, d: int, scale: float = 1.0):

    f = jnp.log((1 / 10000.0)) / (d // 2 - 1)
    f = scale * jnp.exp(f * jnp.arange(d // 2))[jnp.newaxis, ...]

    y = jnp.empty(x.shape + (d, ))
    y = y.at[..., 0::2].set(jnp.sin(x[..., jnp.newaxis] * f))
    y = y.at[..., 1::2].set(jnp.cos(x[..., jnp.newaxis] * f))
    y = y.reshape(*y.shape[:-2], -1)

    return y


@modelclass
class MLP:

    layers: Dynamic[Linear]
    x_pos_dim: int
    t_pos_dim: int

    def __init__(self, x_pos_dim: int, t_pos_dim: int, inner_dim: int, out_dim: int, n_layers: int, key):

        keys = jax.random.split(key, n_layers + 2)
        self.layers = []
        
        self.layers += [Linear(x_pos_dim + t_pos_dim, inner_dim, keys[0])]
        self.layers += [Linear(inner_dim, inner_dim, keys[i]) for i in range(1, n_layers + 1)]
        self.layers += [Linear(inner_dim, out_dim, keys[-1])]

        self.x_pos_dim = x_pos_dim
        self.t_pos_dim = t_pos_dim

    def __call__(self, x, t):

        x = sinusoidal_emb(x, self.x_pos_dim // x.shape[-1], scale=25.0)
        t = sinusoidal_emb(t, self.t_pos_dim)

        y = jnp.concatenate((x, t), -1)

        for i, layer in enumerate(self.layers):
            y = layer(y)

            if i != len(self.layers) - 1:
                y = jax.nn.relu(y)

        return y
