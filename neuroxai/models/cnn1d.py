import torch
import torch.nn as nn

class EEG_CNN(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.fc(x).squeeze(-1)
