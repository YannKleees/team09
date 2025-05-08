# File: src/models.py

import torch.nn as nn

################################################################################
# Single-branch CNN for EITHER binary or multi-class
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
# Two‐headed CNN (binary + multi)
################################################################################

class DoubleHeadCNN(nn.Module):
    """
    A CNN that uses a single shared feature extractor but has two output "heads":
      1) Binary: no_tumor vs tumor
      2) Multi: 3 tumor subtypes (or 4 if needed)
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

        self.head_binary = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # no_tumor vs tumor
        )

        self.head_multi = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)  # or 4 if you want 3 subtypes + notumor
        )

    def forward(self, x, mode='binary'):
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
    Create a model instance based on model_variant:
       "binary", "multi", "double", "simple", "transfer", or "custom".

    Returns:
       model (nn.Module), num_classes (None if 'double')
    """

    if model_variant == "binary":
        # Single-head CNN for exactly 2 classes
        return SimpleCNN(num_classes=2, input_size=input_size, dropout=dropout), 2

    elif model_variant == "multi":
        # Single-head CNN for multiple classes (set to 4 by default).
        return SimpleCNN(num_classes=4, input_size=input_size, dropout=dropout), 4

    elif model_variant == "double":
        # Two-headed CNN
        return DoubleHeadCNN(input_size=input_size, dropout=dropout), None

    elif model_variant == "simple":
        # "Simple" also uses the same baseline, but let's default to 2 classes
        # (you can adjust to 4 if you prefer).
        return SimpleCNN(num_classes=2, input_size=input_size, dropout=dropout), 2

    elif model_variant == "transfer":
        # TODO: implement your transfer learning architecture
        raise NotImplementedError("Transfer learning model => TODO here.")

    elif model_variant == "custom":
        # TODO: implement your custom advanced model
        raise NotImplementedError("Custom model => TODO here.")

    else:
        raise ValueError(f"Unknown model_variant: {model_variant}")
