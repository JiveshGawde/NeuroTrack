import torch
import torch.nn as nn


class NeuroTrackModel(nn.Module):

    def __init__(self, hidden_channels: list[int]):
        self.hidden_channels: list[int] = hidden_channels
