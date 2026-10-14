import os

import torch


def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir="checkpoints"):
    """Save model + optimizer state and metrics to checkpoint_dir/model_epoch_XXX.pth."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"model_epoch_{epoch:03d}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
        },
        path,
    )
    return path


def load_checkpoint(model, optimizer, checkpoint_path, device="cpu"):
    """Restore model (and optimizer, if given). Returns (epoch, loss, accuracy).

    The optimizer state matters when resuming training: Adam keeps running
    averages (momentum / variance) per parameter, and without them training
    restarts "cold" and can jump around.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"], checkpoint["accuracy"]
