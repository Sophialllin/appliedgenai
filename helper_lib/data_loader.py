import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# The assignment's CNN expects 64x64 RGB input, but CIFAR10 images are 32x32,
# so we resize. The API must use this SAME transform at inference time.
IMAGE_SIZE = 64


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_data_loader(data_dir, batch_size=32, train=True):
    """CIFAR10 loader. train=True -> 50k training images (shuffled);
    train=False -> 10k test images (not shuffled)."""
    dataset = datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=get_transform()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def get_train_val_loaders(data_dir, batch_size=32, val_ratio=0.1, seed=42):
    """Split the CIFAR10 training set into train / validation loaders."""
    dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=get_transform()
    )
    n_val = int(len(dataset) * val_ratio)
    train_ds, val_ds = random_split(
        dataset,
        [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
