"""
Deep Learning models for RUL prediction on CMAPSS.
- CNN-LSTM hybrid (strong temporal + local feature extractor)
- Pure LSTM baseline
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM for multivariate sensor time-series RUL regression.
    Proven architecture in many CMAPSS papers.
    """

    def __init__(self, n_features: int, sequence_length: int = 30, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # After two conv + pool: sequence_length // 2
        lstm_input_size = 32
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0,
            bidirectional=False,
        )
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # (batch, 32, seq_len//2)
        x = x.permute(0, 2, 1)  # (batch, seq_len//2, 32)
        x = self.dropout(x)

        lstm_out, (h_n, _) = self.lstm(x)
        # Use last hidden state
        out = h_n[-1]  # (batch, 64)
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)
        return out.squeeze(-1)


class LSTMOnly(nn.Module):
    """Simple 2-layer LSTM for comparison."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        return self.fc(out).squeeze(-1)


def get_model(model_name: str, n_features: int, sequence_length: int = 30):
    if model_name.lower() == "cnn_lstm":
        return CNNLSTM(n_features, sequence_length)
    elif model_name.lower() == "lstm":
        return LSTMOnly(n_features)
    else:
        raise ValueError(f"Unknown model: {model_name}")
