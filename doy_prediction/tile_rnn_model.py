from __future__ import annotations

import torch
from torch import nn


class TileRNNRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "gru",
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.in_channels = in_channels

        rnn_cls = resolve_rnn_class(rnn_type)
        self.rnn_type = rnn_type.lower()
        self.rnn = rnn_cls(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        directions = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * directions, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, channels, seq_len), got {tuple(x.shape)}")

        x_seq = x.transpose(1, 2)
        _, hidden = self.rnn(x_seq)
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        num_directions = 2 if self.rnn.bidirectional else 1
        hidden = hidden.view(self.rnn.num_layers, num_directions, x.shape[0], self.rnn.hidden_size)
        last_layer = hidden[-1].transpose(0, 1).reshape(x.shape[0], self.rnn.hidden_size * num_directions)
        return self.head(last_layer)


def resolve_rnn_class(rnn_type: str) -> type[nn.RNNBase]:
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        return nn.GRU
    if rnn_type == "lstm":
        return nn.LSTM
    if rnn_type == "rnn":
        return nn.RNN
    raise ValueError("rnn_type must be one of: gru, lstm, rnn")
