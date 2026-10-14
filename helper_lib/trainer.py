import os

import torch
from tqdm import tqdm

from .checkpoints import save_checkpoint
from .evaluator import evaluate_model


def train_model(model, train_loader, val_loader, criterion, optimizer,
                device="cpu", epochs=10, checkpoint_dir="checkpoints"):
    """Train the model, validate each epoch, checkpoint each epoch, and keep
    the best model (by validation accuracy) at checkpoint_dir/best_model.pth.

    Returns the model with the best weights loaded.
    """
    model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = -1.0
    best_path = os.path.join(checkpoint_dir, "best_model.pth")

    for epoch in range(1, epochs + 1):
        # model.train() must be called every epoch: evaluate_model() switches the
        # model to eval mode, and it would otherwise stay there.
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        progress = tqdm(train_loader, ncols=110, desc=f"Epoch {epoch}/{epochs}")
        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)
            progress.set_postfix(
                loss=f"{running_loss / running_total:.4f}",
                acc=f"{100 * running_correct / running_total:.2f}%",
            )

        train_loss = running_loss / running_total
        train_acc = 100.0 * running_correct / running_total
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}: train loss {train_loss:.4f}, train acc {train_acc:.2f}% | "
            f"val loss {val_loss:.4f}, val acc {val_acc:.2f}%"
        )

        save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_dir)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                    "accuracy": val_acc,
                },
                best_path,
            )

    # Return the best model, not just the last epoch
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    print(f"Best validation accuracy: {best['accuracy']:.2f}% (epoch {best['epoch']})")
    return model
