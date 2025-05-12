import torch.nn as nn

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
# Two‐headed CNN => "double" (binary + threeclass)
################################################################################

class DoubleHeadCNN(nn.Module):
    """
    A CNN that uses a single shared feature extractor but has two output "heads":
      1) Binary: no_tumor vs tumor
      2) 3-class: pituitary / meningioma / glioma
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

        # Head 1: Binary => NoTumor vs Tumor
        self.head_binary = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

        # Head 2: threeclass => Pituitary, Meningioma, Glioma
        self.head_three = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)
        )

    def forward(self, x, mode='binary'):
        feats = self.shared_features(x)
        if mode == 'binary':
            return self.head_binary(feats)
        else:
            return self.head_three(feats)


################################################################################
# get_model with new names: "simple4", "threeclass", "binary", "double"
################################################################################

def get_model(model_variant, input_size=128, dropout=0.0):
    """
    Create a model instance:
      - simple4 => single CNN, 4-class
      - threeclass => single CNN, 3-class
      - binary => single CNN, 2-class
      - double => two-headed net (binary + threeclass)
      - transfer / custom => placeholders
    """
    if model_variant == "simple":
        return SimpleCNN(num_classes=4, input_size=input_size, dropout=dropout), 4

    elif model_variant == "threeclass":
        return SimpleCNN(num_classes=3, input_size=input_size, dropout=dropout), 3

    elif model_variant == "binary":
        return SimpleCNN(num_classes=2, input_size=input_size, dropout=dropout), 2

    elif model_variant == "double":
        return DoubleHeadCNN(input_size=input_size, dropout=dropout), None

    elif model_variant == "transfer":
        raise NotImplementedError("Transfer learning => implement your advanced net here.")
    elif model_variant == "custom":
        raise NotImplementedError("Custom model => implement your advanced net here.")

    else:
        raise ValueError(f"Unknown model variant: {model_variant}")
