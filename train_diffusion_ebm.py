"""
train_diffusion_ebm.py
-----------------------
python train_diffusion_ebm.py --model diffusion --epochs 50
python train_diffusion_ebm.py --model ebm       --epochs 50
"""

import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from helper_lib.model import DiffusionModel, EnergyModel
from helper_lib.trainer import train_diffusion, train_ebm

os.makedirs("trained_models", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def get_cifar10_loader(batch_size=64):
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # → [-1, 1]
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=2, pin_memory=True)


def run_diffusion(epochs, lr, batch_size):
    model     = DiffusionModel(img_channels=3, base_channels=64, time_dim=128, T=1000)
    loader    = get_cifar10_loader(batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = train_diffusion(model, loader, criterion, optimizer,
                            device=DEVICE, epochs=epochs)

    out_path = "trained_models/diffusion_cifar10.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "img_channels": 3, "base_channels": 64,
        "time_dim": 128, "T": 1000,
    }, out_path)
    print(f"Saved → {out_path}")


def run_ebm(epochs, lr, batch_size, langevin_steps, step_size, noise_scale):
    model     = EnergyModel(img_channels=3)
    loader    = get_cifar10_loader(batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = train_ebm(model, loader, criterion=None, optimizer=optimizer,
                      device=DEVICE, epochs=epochs,
                      langevin_steps=langevin_steps,
                      step_size=step_size, noise_scale=noise_scale)

    out_path = "trained_models/ebm_cifar10.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "img_channels": 3,
    }, out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          choices=["diffusion", "ebm"], required=True)
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--lr",             type=float, default=2e-4)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--langevin_steps", type=int,   default=60)
    parser.add_argument("--step_size",      type=float, default=10.0)
    parser.add_argument("--noise_scale",    type=float, default=0.005)
    args = parser.parse_args()

    if args.model == "diffusion":
        run_diffusion(args.epochs, args.lr, args.batch_size)
    else:
        run_ebm(args.epochs, args.lr, args.batch_size,
                args.langevin_steps, args.step_size, args.noise_scale)