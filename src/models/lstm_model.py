import torch
import torch.nn as nn

class RehabLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=10, num_layers=2):
        super(RehabLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)
