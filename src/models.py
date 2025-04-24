"""
models.py

Defines all model architectures used in this project,
including a simple single-branch CNN (SimpleCNN) and
a double-headed CNN (DoubleHeadCNN). Easily extensible
to new architectures in the future.
"""

import torch.nn as nn


################################################################################
# Single-branch CNN for EITHER binary or multi-class
################################################################################
class SimpleCNN(nn.Module):
    """
    A baseline CNN with optional dropout.
    - num_classes: number of output classes
       e.g. 2 for (tumor vs. no tumor), or 4 for multiple tumor types
    - input_size: input image dimension (assumes square)
    - dropout: dropout probability

    Example usage:
        model = SimpleCNN(num_classes=2, input_size=128, dropout=0.3)
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
        # if input_size=128, after two MaxPool(2) we get (128/4 = 32),
        # so flattened_dim = channels * 32 * 32
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
# Two‐headed CNN (binary + multi)
################################################################################
class DoubleHeadCNN(nn.Module):
    """
    A CNN that uses a single shared feature extractor but has two output "heads":
      1) A binary head for (no_tumor vs. tumor)
      2) A multi-class head (e.g. 3 tumor subtypes)

    The forward pass picks which head to use by specifying mode='binary' or 'multi'.

    Example usage:
        model = DoubleHeadCNN(input_size=224, dropout=0.1)
        # model(x, mode='binary') => shape [N,2]
        # model(x, mode='multi')  => shape [N,3]
    """
    def __init__(self, input_size=128, dropout=0.0):
        super().__init__()
        # Shared feature extraction layers
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

        # Head 1: Binary classification
        self.head_binary = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # 2 classes
        )

        # Head 2: Multi-class classification
        self.head_multi = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)  # 3 classes for tumor subtypes
        )

    def forward(self, x, mode='binary'):
        # Extract shared features
        feats = self.shared_features(x)
        if mode == 'binary':
            return self.head_binary(feats)
        else:
            return self.head_multi(feats)


################################################################################
# Helper to retrieve a chosen model
################################################################################
def get_model(model_variant, input_size=128, dropout=0.0):
    """
    Create a model instance based on the model_variant string.
    Returns:
        model (nn.Module), num_classes (int or None if double)

    model_variant can be one of:
       - 'binary' => SimpleCNN(num_classes=2)
       - 'multi'  => SimpleCNN(num_classes=4)
       - 'double' => DoubleHeadCNN()
       - (You can add more custom variants as needed.)

    Example usage:
        model, nc = get_model("binary", input_size=128, dropout=0.2)
    """
    if model_variant == "binary":
        # 2-class classification (no tumor vs tumor)
        return SimpleCNN(num_classes=2, input_size=input_size, dropout=dropout), 2

    elif model_variant == "multi":
        # 4-class classification (no tumor + 3 tumor subtypes)
        return SimpleCNN(num_classes=4, input_size=input_size, dropout=dropout), 4

    elif model_variant == "double":
        # Double-headed CNN
        # For 'double', the number of classes is not a single integer,
        # since we effectively have 2 heads with 2-class and 3-class outputs.
        return DoubleHeadCNN(input_size=input_size, dropout=dropout), None

    else:
        raise ValueError(f"Unknown model_variant: {model_variant}")
