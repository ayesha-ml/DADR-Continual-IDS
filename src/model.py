import torch
import torch.nn as nn
from src.config import Config


class ContinualMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        # initializing shared feature extractor and expanding classification head
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        hidden_1, hidden_2 = Config.HIDDEN_DIMS
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )
        self.classifier = nn.Linear(hidden_2, num_classes)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        # extracting 128-dimensional latent representations before final logits
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # computing class logits from input network flow features
        features = self.get_embeddings(x)
        return self.classifier(features)

    def expand_classes(self, new_total_classes: int):
        # expanding output head when new attack classes appear while preserving existing weights
        if new_total_classes <= self.num_classes:
            return

        device = next(self.parameters()).device
        in_features = self.classifier.in_features
        new_classifier = nn.Linear(in_features, new_total_classes).to(device)

        with torch.no_grad():
            new_classifier.weight[: self.num_classes] = self.classifier.weight.data
            new_classifier.bias[: self.num_classes] = self.classifier.bias.data

        self.classifier = new_classifier
        self.num_classes = new_total_classes
