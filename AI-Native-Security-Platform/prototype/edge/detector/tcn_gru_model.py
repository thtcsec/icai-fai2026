"""
tcn_gru_model.py - Hybrid TCN-GRU Autoencoder & Classifier for Edge Telemetry

Grounded implementation ported from sdn-its-resilience-ai (IEDE Autumn project).
Combines Temporal Convolutional Networks (TCN) for multi-scale feature extraction,
a GRU Autoencoder for reconstruction error calculation RE(x) = ||x - x_hat||_2^2,
and a Softmax classification head for threat taxonomy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any

class Chomp1d(nn.Module):
    """Trims trailing padding to enforce causal temporal convolutions."""
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """Residual Block in Temporal Convolutional Network (TCN)."""
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, 
                 dilation: int, padding: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Multi-layer Dilated Temporal Convolutional Network."""
    def __init__(self, num_inputs: int, num_channels: List[int], kernel_size: int = 3, dropout: float = 0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                              dilation=dilation_size, padding=padding, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GRUAutoencoder(nn.Module):
    """Encoder-Decoder GRU for Sequence Reconstruction."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super(GRUAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.reconstruct_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        _, hidden = self.encoder(x)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(decoder_input, hidden)
        x_hat = self.reconstruct_head(dec_out)
        return x_hat, hidden[-1]


class TCNGRUResilienceModel(nn.Module):
    """
    Hybrid TCN-GRU network resilience model.

    Outputs:
      1. x_hat: reconstructed sequence (auxiliary representation objective)
      2. logits: classification head

    The default ``num_classes=6`` head is a legacy multi-way layout
    (Normal/DDoS/Probe/Botnet/Fault/LOFT). The training harness used in this
    artifact collapses labels to binary normal-vs-attack and scores
    ``1 - P(normal)``; logits for classes 2-5 are therefore unsupervised in the
    reported experiments. Keep ``num_classes=6`` for checkpoint compatibility;
    do not interpret reported F1 as a six-way taxonomy result.
    """
    def __init__(self, num_features: int = 10, num_classes: int = 6, tcn_channels: List[int] = [16, 32], hidden_dim: int = 32):
        super(TCNGRUResilienceModel, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.tcn = TemporalConvNet(num_inputs=num_features, num_channels=tcn_channels)
        self.autoencoder = GRUAutoencoder(input_dim=tcn_channels[-1], hidden_dim=hidden_dim)
        
        self.proj_back = nn.Linear(tcn_channels[-1], num_features)
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: [batch_size, seq_len, num_features]
        batch_size, seq_len, _ = x.shape
        
        # TCN expects [batch_size, num_features, seq_len]
        x_tcn_in = x.transpose(1, 2)
        x_tcn_out = self.tcn(x_tcn_in).transpose(1, 2) # back to [batch_size, seq_len, channels]
        
        # GRU Autoencoder
        reconstructed_tcn, latent = self.autoencoder(x_tcn_out)
        x_hat = self.proj_back(reconstructed_tcn)
        
        # Classification logits
        logits = self.classifier_head(latent)
        
        # Reconstruction Error RE(x)
        rec_error = torch.mean((x - x_hat) ** 2, dim=(1, 2))
        
        return x_hat, logits, rec_error
