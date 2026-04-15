import torch
import torch.nn as nn


class NeuroTrackNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden_channels: nn.ModuleList = []

    def forward(self):
        ...
