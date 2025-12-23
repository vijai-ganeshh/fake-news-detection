import torch
import torch.nn as nn

class FusionClassifier(nn.Module):
    def __init__(self, input_dim=896):  # 768 + 128
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)
