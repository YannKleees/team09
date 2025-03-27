import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import BrainTumorDataset
from src.transforms_config import create_transform

class SimpleCNN(nn.Module):
    """
    Example CNN outline for your future training. 
    Replace with any architecture you like.
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16* (128//4)* (128//4), 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Experiment:
    """
    Placeholder class to demonstrate how you might structure
    a training/evaluation pipeline for your cleaned dataset.
    """
    def __init__(
        self,
        data_root="cleaned_data",
        batch_size=8,
        lr=1e-3,
        epochs=1
    ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        # create a transform for training or do real-time augmentation
        self.transform = create_transform(
            image_size=(128,128),
            normalization="zscore",
            augment=True  # or False if you want
        )

        self.model = SimpleCNN(num_classes=4)  # example
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def load_data(self):
        # For example, we read "Training" and "Testing" from cleaned_data
        train_set = BrainTumorDataset(
            root_dir=self.data_root, subset="Training", transform=self.transform
        )
        val_set   = BrainTumorDataset(
            root_dir=self.data_root, subset="Testing", transform=self.transform
        )
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

    def train_one_epoch(self):
        self.model.train()
        for images, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        print("One epoch done.")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        for images, labels in self.val_loader:
            with torch.no_grad():
                outputs = self.model(images)
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = 100.0 * correct / total
        print(f"Validation accuracy: {accuracy:.2f}%")
        return accuracy

    def run_experiment(self):
        self.load_data()
        for ep in range(self.epochs):
            self.train_one_epoch()
            acc = self.evaluate()
            print(f"Epoch {ep+1}/{self.epochs}, val acc = {acc:.2f}%")
