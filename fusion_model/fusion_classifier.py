import torch
import torch.nn as nn


class FusionClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super(FusionClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),  # classifier.0
            nn.ReLU(),                  # classifier.1

            nn.Identity(),              # classifier.2  <-- IMPORTANT FIX

            nn.Linear(256, 1),           # classifier.3
            nn.Sigmoid()                 # classifier.4
        )

    def forward(self, x):
        return self.classifier(x)
