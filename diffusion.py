import jax
import jax.numpy as jnp
from jax_primitives import MLP, Adam
from functools import partial
import numpy as np


class DDPM:

    def __init__(
        self,
        model,
        data,
        steps,
        data_shape,
        batch_size=512,
        betas=(1e-4, 0.02)
    ):
        self.model = model
        self.data = data
        self.data_shape = data_shape

        self.steps = steps
        self.beta = jnp.linspace(betas[0], betas[1], steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = jnp.cumprod(self.alpha, axis=0)
        
        self.epoch = 0
        self.batch_size = batch_size

        self.opt = Adam(self.model, alpha=1e-4)

    def run_epoch(self, key):
        loss_history = []

        def mse(model, x, t, e):
            return jnp.mean((model(x, t) - e) ** 2)
        
        @jax.jit
        def update(opt, model, x, key):
            key_time, key_noise = jax.random.split(key, 2)

            t = jax.random.randint(key_time, [len(x)] + (len(x.shape) - 1) * [1], 0, self.steps)
            e = jax.random.normal(key_noise, x.shape)
            x_e = jnp.sqrt(self.alpha_bar[t]) * x + jnp.sqrt(1 - self.alpha_bar[t]) * e

            loss, grads = jax.value_and_grad(mse)(model, x_e, t, e)

            model = opt.step(model, grads)

            return opt, model, loss

        for i in range(0, len(self.data), self.batch_size):
            x = self.data[i: i + self.batch_size]
            self.opt, self.model, loss = update(self.opt, self.model, x, key)
            loss_history.append(loss)

        return np.array(loss_history).mean().item()


    def sample(self, key, n: int = 64, return_all_steps: bool = False):
        keys = jax.random.split(key, self.steps)

        x = jax.random.normal(keys[0], [n] + self.data_shape)

        if return_all_steps:
            steps = []

        @jax.jit
        def sample_step(model, x, z, i):
            t = jnp.ones([n] + (len(x.shape) - 1) * [1], dtype=jnp.uint16) * i
                
            a = self.alpha[t]
            a_bar = self.alpha_bar[t]

            x = x - (1 - a) / jnp.sqrt(1 - a_bar) * model(x, t)
            x = x / jnp.sqrt(a) + jnp.sqrt(self.beta[t]) * z

            return x
        
        for i in reversed(range(self.steps)):
            if i > 0:
                z = jax.random.normal(keys[i], x.shape)
            else:
                z = jnp.zeros(x.shape)

            x = sample_step(self.model, x, z, i)

            if return_all_steps:
                steps.append(x)

            if return_all_steps:
                return steps
            else:
                return x