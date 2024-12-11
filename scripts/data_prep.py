
import torchvision
import torchvision.transforms as transforms
import os

def download_data(data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_data = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    val_data = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    return train_data, val_data

if __name__ == "__main__":
    download_data()
