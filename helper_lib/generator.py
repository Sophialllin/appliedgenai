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
