"""Train the assignment CNN on CIFAR10 and report test accuracy.

Usage:  python train.py --epochs 10
Output: checkpoints/best_model.pth (this is the file the API loads)
"""
import argparse

import torch.nn as nn
import torch.optim as optim

from helper_lib import (
    get_data_loader,
    get_train_val_loaders,
    get_model,
    train_model,
    evaluate_model,
    get_device,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = get_train_val_loaders(args.data_dir, args.batch_size)
    test_loader = get_data_loader(args.data_dir, args.batch_size, train=False)

    model = get_model("CNN")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        device=device, epochs=args.epochs, checkpoint_dir=args.checkpoint_dir,
    )

    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
