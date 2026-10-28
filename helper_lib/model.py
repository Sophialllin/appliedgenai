import torch.nn as nn


class AssignmentCNN(nn.Module):
    """CNN from Assignment 2.

    Input:   3 x 64 x 64
    Conv2d(3 -> 16, k=3, s=1, p=1) + ReLU   -> 16 x 64 x 64
    MaxPool2d(2, 2)                          -> 16 x 32 x 32
    Conv2d(16 -> 32, k=3, s=1, p=1) + ReLU  -> 32 x 32 x 32
    MaxPool2d(2, 2)                          -> 32 x 16 x 16
    Flatten                                  -> 8192
    Linear(8192 -> 100) + ReLU               -> 100
    Linear(100 -> 10)                        -> 10
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def get_model(model_name="CNN"):
    """Return a model by name. 'CNN' is the assignment architecture."""
    if model_name in ("CNN", "AssignmentCNN"):
        return AssignmentCNN()
    raise ValueError(f"Unknown model: {model_name!r}. Available: 'CNN'")


import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    GAN Generator.

    Architecture (from assignment3):
      Input : noise vector (BATCH_SIZE, 100)
      FC    : 100 → 7*7*128, reshape to (BATCH_SIZE, 128, 7, 7)
      ConvT : 128 → 64,  k=4, s=2, p=1  →  14×14  + BN + ReLU
      ConvT :  64 →  1,  k=4, s=2, p=1  →  28×28  + Tanh
    """

    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.conv_blocks = nn.Sequential(
            # 128 → 64, 7×7 → 14×14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 → 1, 14×14 → 28×28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)               # (B, 7*7*128)
        x = x.view(-1, 128, 7, 7)   # (B, 128, 7, 7)
        return self.conv_blocks(x)   # (B, 1, 28, 28)


class Discriminator(nn.Module):
    """
    GAN Discriminator.

    Architecture (from assignment3):
      Input : image (1, 28, 28)
      Conv  :   1 →  64, k=4, s=2, p=1  →  14×14  + LeakyReLU(0.2)
      Conv  :  64 → 128, k=4, s=2, p=1  →   7×7   + BN + LeakyReLU(0.2)
      Flatten → Linear(128*7*7, 1)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 1 → 64, 28×28 → 14×14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 → 128, 14×14 → 7×7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # classify
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 1) — raw logit, no sigmoid


class GAN(nn.Module):
    """Container that holds both Generator and Discriminator."""

    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()



# Factory used by the rest of the helper library


def get_model(model_name="CNN"):
    if model_name in ("CNN", "AssignmentCNN"):
        return AssignmentCNN()
    if model_name == "GAN":
        return GAN()
    raise ValueError(f"Unknown model: {model_name!r}. Available: 'CNN', 'GAN'")