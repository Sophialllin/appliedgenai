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
