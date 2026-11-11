import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def generate_samples(model, device, num_samples: int = 10):
    """Generate and display samples from a trained GAN generator.

    Args:
        model      : GAN instance with a .generator and .latent_dim attribute.
        device     : 'cpu' or 'cuda'.
        num_samples: Number of images to generate.
    """
    model.generator.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim, device=device)
        samples = model.generator(z)   # (num_samples, 1, 28, 28) in [-1, 1]

    nrow = min(num_samples, 5)
    grid = vutils.make_grid(
        samples, nrow=nrow, normalize=True, value_range=(-1, 1)
    )

    plt.figure(figsize=(nrow * 1.5, (num_samples // nrow + 1) * 1.5))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.title("Generated MNIST Digits")
    plt.tight_layout()
    plt.show()


def generate_diffusion_samples(model, device, num_samples=10, diffusion_steps=100):
    """
    DDPM reverse diffusion: start from pure Gaussian noise, 
    iteratively denoise over `diffusion_steps` steps.
    Returns a tensor of shape (num_samples, 3, 32, 32) in [-1, 1].
    """
    import torch

    model.to(device).eval()
    T = model.T

    # Pick evenly-spaced timesteps from T-1 down to 0
    step_indices = list(range(0, T, max(1, T // diffusion_steps)))[::-1]

    with torch.no_grad():
        x = torch.randn(num_samples, 3, 32, 32, device=device)

        for i, t_val in enumerate(step_indices):
            t         = torch.full((num_samples,), t_val,
                                   dtype=torch.long, device=device)
            eps_pred  = model(x, t)

            alpha     = model.alphas[t][:, None, None, None]
            alpha_bar = model.alpha_bars[t][:, None, None, None]
            beta      = model.betas[t][:, None, None, None]

            # DDPM reverse step:
            # x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-ā_t) * eps_pred)
            #           + sqrt(beta_t) * z   (z=0 on last step)
            coeff  = beta / (1.0 - alpha_bar).sqrt()
            x_mean = (1.0 / alpha.sqrt()) * (x - coeff * eps_pred)

            if i < len(step_indices) - 1:
                x = x_mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = x_mean   # last step: no noise added

    return x.clamp(-1, 1)


def generate_ebm_samples(model, device, num_samples=10,
                         n_steps=100, step_size=10.0, noise_scale=0.005):
    """
    Langevin dynamics: start from uniform noise, iteratively update
    the INPUT IMAGE x (not model weights) to minimise energy E(x).
    
    Key: x.requires_grad_(True) is set on the IMAGE tensor,
         so grad flows through the energy CNN back to x, not to model params.
    """
    import torch

    model.to(device).eval()

    # Start from random noise in [-1, 1]
    x = torch.rand(num_samples, 3, 32, 32, device=device) * 2 - 1
    x.requires_grad_(True)   # ← grad w.r.t.
    for _ in range(n_steps):
        if x.grad is not None:
            x.grad.zero_()
        energy = model(x).sum()
        energy.backward()
        with torch.no_grad():
            x -= (step_size / 2) * x.grad
            x += noise_scale * torch.randn_like(x)
            x.clamp_(-1.0, 1.0)
        x = x.detach().requires_grad_(True)
    return x.detach().clamp(-1, 1)