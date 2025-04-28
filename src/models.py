"""
models.py

Defines several model architectures for the Brain Tumor MRI project:
  - SimpleCNN for binary or multi-class classification
  - DoubleHeadCNN for a two-headed approach (binary + multi)
  - Transfer Learning placeholder
  - Custom advanced model placeholder

Easily extensible to new architectures in the future.
"""

import torch.nn as nn


################################################################################
# Single-branch CNN for EITHER binary or multi-class
################################################################################
class SimpleCNN(nn.Module):
    """
    A baseline CNN with optional dropout.
    - num_classes: number of output classes
       e.g. 2 for (tumor vs. no tumor) OR 4 for multiple tumor types (0=No,1=Meningioma,2=Glioma,3=Pituitary)
         * If you only have 3 tumor subtypes (and no 'no tumor'), set num_classes=3.
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
        # If input_size=128, after two MaxPool(2) we get 128 -> 64 -> 32,
        # so flattened_dim = 16 * (input_size//4) * (input_size//4)
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
      2) A multi-class head (e.g. 3 tumor subtypes or 4 if you included "no tumor" there)

    The forward pass picks which head to use by specifying mode='binary' or 'multi'.

    Example usage:
        model = DoubleHeadCNN(input_size=224, dropout=0.1)
        # model(x, mode='binary') => shape [N,2]
        # model(x, mode='multi')  => shape [N,3] or [N,4] if you changed that layer
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
            nn.Linear(32, 2)  # 2 classes: no tumor vs tumor
        )

        # Head 2: Multi-class classification (default 3, but you can adjust to 4)
        self.head_multi = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)  # By default, 3 classes for tumor subtypes
        )

    def forward(self, x, mode='binary'):
        # Extract shared features
        feats = self.shared_features(x)
        if mode == 'binary':
            return self.head_binary(feats)
        else:
            return self.head_multi(feats)


################################################################################
# Placeholder: Transfer Learning approach
################################################################################
# TODO: Implement a transfer learning model, e.g., using torchvision.models.resnet18
#       or any other pre-trained architecture. The skeleton might look like:
#
# import torchvision.models as models
#
# class TransferModel(nn.Module):
#     def __init__(self, num_classes=2, dropout=0.0):
#         super().__init__()
#         # TODO: load a pretrained model, replace the classifier head
#         # e.g. self.backbone = models.resnet18(pretrained=True)
#         # freeze or unfreeze layers, etc.
#         # self.fc = nn.Linear(512, num_classes)  # for example
#         # ...
#     def forward(self, x):
#         # TODO: define forward pass
#         pass


################################################################################
# Placeholder: Custom advanced architecture
################################################################################
# TODO: Implement a more complex architecture (multi-scale, attention, etc.).
# For example:
#
# class CustomAdvancedCNN(nn.Module):
#     def __init__(self, ...):
#         super().__init__()
#         # define your layers
#     def forward(self, x):
#         # define forward pass
#         pass


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
       - 'multi'  => SimpleCNN(num_classes=4) (by default)
       - 'double' => DoubleHeadCNN() (2-head approach)
       - 'transfer' => TODO: implement
       - 'custom'   => TODO: implement

    Example usage:
        model, nc = get_model("binary", input_size=128, dropout=0.2)
    """
    if model_variant == "binary":
        # 2-class classification (no tumor vs tumor)
        return SimpleCNN(num_classes=2, input_size=input_size, dropout=dropout), 2

    elif model_variant == "multi":
        # 4-class classification by default (0=No tumor + 3 subtypes)
        # If you have only 3 subtypes, modify in SimpleCNN or pass num_classes=3.
        return SimpleCNN(num_classes=4, input_size=input_size, dropout=dropout), 4

    elif model_variant == "double":
        # Double-headed CNN
        # For 'double', the number of classes is not a single integer,
        # since we effectively have 2 heads: [2-class] + [3 or 4 class].
        return DoubleHeadCNN(input_size=input_size, dropout=dropout), None

    elif model_variant == "transfer":
        # TODO: Implement a real TransferModel once you define it
        raise NotImplementedError("Transfer learning model not yet implemented. See TODO in models.py")

    elif model_variant == "custom":
        # TODO: Implement a real CustomAdvancedCNN once you define it
        raise NotImplementedError("Custom advanced model not yet implemented. See TODO in models.py")

    else:
        raise ValueError(f"Unknown model_variant: {model_variant}")
