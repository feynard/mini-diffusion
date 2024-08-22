# Simple DDPM Scheduler

Inspired by [tiny-diffusion](https://github.com/tanelp/tiny-diffusion)

Example of usage:

```Python
time_dim = 128
data_dim = 3
pos_dim = 32 * data_dim
diffusion_steps = 512
n_epochs = 10_000

k = jp.RandomKey(0) # import jax_primitives as jp

net = MLP(pos_dim, time_dim, 256, data_dim, 6, k // 1)

diffusion = DDPM(
    model=net,
    data=data,
    steps=diffusion_steps,
    data_shape=(data_dim, ),
    batch_size=8192,
    betas=(1e-4, 0.02)
)

for _ in range(n_epochs):
    diffusion.run_epoch(k // 1)

y = diffusion.sample(k, 10_000)
```

## Fitting a 3D Point Cloud

<p align="center">
  <img src="images/example.png" alt="Point Cloud"/>
</p>

Source: [Point Cloud of Roses (Sketchfab)](https://sketchfab.com/3d-models/a-point-cloud-of-roses-13495f32fe1340fe91fd35a42c0a76c3)
