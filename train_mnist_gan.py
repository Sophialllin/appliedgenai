"""
train_mnist_gan.py
------------------
Train the GAN on MNIST and save the weights to trained_models/gan_mnist.pth.

Usage:
    python train_mnist_gan.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE  = 64
EPOCHS      = 20
LR          = 2e-4
LATENT_DIM  = 100
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH   = "trained_models/gan_mnist.pth"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),   # maps [0,1] → [-1,1]
])

train_dataset = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

# ---------------------------------------------------------------------------
# Model, loss, optimizers
# ---------------------------------------------------------------------------
model = get_model("GAN")

criterion = nn.BCEWithLogitsLoss()

optimizer = {
    "G": torch.optim.Adam(model.generator.parameters(),     lr=LR, betas=(0.5, 0.999)),
    "D": torch.optim.Adam(model.discriminator.parameters(), lr=LR, betas=(0.5, 0.999)),
}

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
print(f"Training on {DEVICE} for {EPOCHS} epochs …")
model = train_gan(model, train_loader, criterion, optimizer, device=DEVICE, epochs=EPOCHS)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs("trained_models", exist_ok=True)
torch.save(
    {
        "generator":     model.generator.state_dict(),
        "discriminator": model.discriminator.state_dict(),
        "latent_dim":    LATENT_DIM,
    },
    SAVE_PATH,
)
print(f"Saved model to {SAVE_PATH}")

# ---------------------------------------------------------------------------
# Quick visual check
# ---------------------------------------------------------------------------
generate_samples(model, device=DEVICE, num_samples=10)
