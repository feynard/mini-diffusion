import matplotlib.pyplot as plt
import numpy as np
from jax_primitives import RandomKey
from tqdm import tqdm

from diffusion import DDPM
from mlp import MLP


if __name__ == '__main__':

    n = 16_384

    r = np.linspace(2 * np.pi, 8 * np.pi, n)
    e = np.random.normal(0, 0.25, n)

    x = (r + e) / (6 * np.pi) * np.cos(r)
    y = (r + e) / (6 * np.pi) * np.sin(r)

    x = np.stack((x, y), 1)
    x /= 2

    plt.figure(figsize=(8, 8))
    plt.axis('equal')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.scatter(x[:, 0], x[:, 1], marker='.')
    plt.savefig('data_original.png')

    n_epochs = 10_000
    t_pos_dim = 128
    x_pos_dim = 64
    diffusion_steps = 512
    data_shape = (2,)

    key = RandomKey(0)
    net = MLP(x_pos_dim * 2, t_pos_dim, 128, 2, 4, key // 1)
    ddpm = DDPM(net, x, diffusion_steps, data_shape)

    for char in (pbar := tqdm(range(n_epochs))):
        average_loss = ddpm.run_epoch(key // 1)
        pbar.set_description(f"Loss {average_loss:.6f}")

    z = ddpm.sample(key // 1, n=8192)
    plt.figure(figsize=(8, 8))
    plt.axis('equal')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.scatter(z[:, 0], z[:, 1], marker='.')
    plt.savefig('data_predicted.png')
