from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)  # Only 2 classes: cat and dog

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Filter CIFAR-10 dataset for cats and dogs
def filter_cifar10(dataset, target_classes):
    indices = [i for i, label in enumerate(dataset.targets) if label in target_classes]
    return Subset(dataset, indices)

def train_model():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Load CIFAR-10
    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)

    # Filter dataset for cats (3) and dogs (5)
    target_classes = [3, 5]  # Cat and Dog labels
    filtered_dataset = filter_cifar10(dataset, target_classes)
    train_loader = DataLoader(filtered_dataset, batch_size=32, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(5):  # Training for 5 epochs
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            # Map labels to [0, 1] for cats and dogs
            labels = torch.tensor([0 if label == 3 else 1 for label in labels])

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "model/cat_dog_model.pth")
    print("Model saved to model/cat_dog_model.pth")

if __name__ == "__main__":
    train_model()
