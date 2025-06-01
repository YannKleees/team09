import torch
import torch.nn as nn
from src.lpconv import LpCNN
################################################################################
# Single-branch CNN for any # of classes
################################################################################

class SimpleCNN(nn.Module):
    """
    A baseline CNN with optional dropout.
    - num_classes: number of output classes
    - input_size: input image dimension (square)
    - dropout: dropout probability
    """
    def __init__(self, num_classes=4, input_size=128, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        flattened_dim = 16 * (input_size // 4) * (input_size // 4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

################################################################################
# Two‐headed CNN => "double" (binary + 3-class)
################################################################################

class DoubleHeadCNN(nn.Module):
    """
    A CNN that uses a single shared feature extractor but has two output "heads":
      (1) Binary: NoTumor vs Tumor
      (2) Threeclass: Pituitary / Meningioma / Glioma

    We choose the output by calling forward(..., mode='binary') or mode='threeclass'.
    """
    def __init__(self, input_size=128, dropout=0.0):
        super().__init__()
        self.shared_features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        flattened_dim = 16 * (input_size // 4) * (input_size // 4)

        # Head 1: Binary => [NoTumor, Tumor]
        self.head_binary = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # => 2 outputs
        )

        # Head 2: threeclass => [Pituitary, Meningioma, Glioma]
        self.head_threeclass = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)
        )

    def forward(self, x, mode='binary'):
        """
        mode='binary' => returns [batch_size, 2] (NoTumor vs Tumor)
        mode='threeclass' => returns [batch_size, 3] (pituitary / meningioma / glioma)
        """
        feats = self.shared_features(x)
        if mode == 'binary':
            return self.head_binary(feats)
        elif mode == 'threeclass':
            return self.head_threeclass(feats)
        else:
            raise ValueError(f"Invalid mode='{mode}' in DoubleHeadCNN.")

################################################################################
# get_model with new names: "simple" (4-class), "threeclass", "binary", "double"
################################################################################

def get_model(model_variant, input_size=128, dropout=0.0):
    """
    Create a model instance:
      - "simple" => single CNN, 4-class
      - "threeclass" => single CNN, 3-class
      - "binary" => single CNN, 2-class
      - "double" => two-headed net (binary + multi)
    """
    if model_variant == "simple":
        return SimpleCNN(num_classes=4, input_size=input_size, dropout=dropout), 4

    elif model_variant == "threeclass":
        return SimpleCNN(num_classes=3, input_size=input_size, dropout=dropout), 3

    elif model_variant == "binary":
        return SimpleCNN(num_classes=2, input_size=input_size, dropout=dropout), 2

    elif model_variant == "double":
        return DoubleHeadCNN(input_size=input_size, dropout=dropout), None
    
    elif model_variant == "lpconv":
        return LpCNN(num_classes=4,input_size=input_size,dropout=dropout,in_channels=1), 4
    
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")
