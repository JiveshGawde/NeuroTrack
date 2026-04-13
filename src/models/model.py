from typing import Literal, Optional
import torch.nn as nn

DEFAULT_KERNEL = 3
DEFAULT_STRIDE = 2
DEFAULT_PADDING = 0
DEFAULT_DROPOUT_P = 0.5
DEFAULT_POOL_K = 2


class NeuroTrackModel(nn.Module):

    def __init__(self, input_channel: int, hidden_channels: list[int], output_channel: int, classes: int,
                 stride: Optional[list[int | None]] = None,
                 kernel: Optional[list[int | None]] = None,
                 dropout: Optional[list[float | None]] = None,
                 padding: Optional[list[int | None]] = None,
                 pools: Optional[list[int | None]] = None,
                 pool_type: Literal['max'] | Literal['avg'] = "max"):
        super().__init__()
        assert len(hidden_channels) + 1 == len(
            stride) if stride is not None else True, "stride should have the same length as hidden channels + 1"
        assert len(hidden_channels) + 1 == len(
            kernel) if kernel is not None else True, "kernel should have the same length as hidden channels + 1"
        assert len(hidden_channels) + 1 == len(
            dropout) if dropout is not None else True, "dropout should have the same length as hidden channels + 1"
        assert len(hidden_channels) + 1 == len(
            padding) if padding is not None else True, "padding should have the same length as hidden channels + 1"
        assert len(hidden_channels) + 1 == len(
            pools) if pools is not None else True, "pools should have the same length as hidden channels + 1"

        self.hidden_channels: list[int] = hidden_channels
        self.input_channel: int = input_channel
        self.output_channel: int = output_channel

        hidden_channels1 = [input_channel] + hidden_channels
        layers = []

        for i in range(1, len(hidden_channels1)):
            kernel_size = DEFAULT_KERNEL
            stride_size = DEFAULT_STRIDE
            dropout_p = DEFAULT_DROPOUT_P
            pool_k = DEFAULT_POOL_K
            padding_size = DEFAULT_PADDING

            if stride is not None and stride[i-1] is not None:
                stride_size = stride[i - 1]

            if kernel is not None and kernel[i - 1] is not None:
                kernel_size = kernel[i - 1]

            if padding is not None and padding[i - 1] is not None:
                padding_size = padding[i - 1]

            if dropout is not None and dropout[i - 1] is not None:
                dropout_p = dropout[i - 1]

            if pools is not None and pools[i - 1] is not None:
                pool_k = pools[i - 1]
            layers.append(MiniNeuroBlock(input_channel=hidden_channels1[i-1],
                                         output_channel=hidden_channels1[i],
                                         kernel_size=kernel_size,
                                         stride=stride_size,
                                         padding=padding_size,
                                         pool_k=pool_k,
                                         dropout_p=dropout_p, pool_type=pool_type))

        self.network_layers = nn.ModuleList(layers)

        self.flatten = nn.Flatten()

        self.latent_space = nn.Linear(hidden_channels[-1], 4096)

        self.latent_space2 = nn.Linear(4096, 1024)

        self.last = nn.Linear(1024, classes)

        self.classifier = nn.Softmax(classes)

    def forward(self, X):

        x = X

        for layer in self.network_layers:
            x = layer(x)
        x = self.flatten(x)

        x = self.latent_space(x)

        x = self.latent_space2(x)

        x = self.last(x)

        x = self.classifier(x)

        return x


class MiniNeuroBlock(nn.Module):

    def __init__(self, input_channel: int,
                 output_channel: int,
                 kernel_size: int = DEFAULT_KERNEL,
                 stride: int = DEFAULT_STRIDE,
                 padding: int = DEFAULT_PADDING,
                 dropout_p: float = DEFAULT_DROPOUT_P,
                 pool_k: int = DEFAULT_POOL_K,
                 pool_type: Literal['max'] | Literal['avg'] = 'max'):

        super().__init__()
        match pool_type:
            case 'max':
                self.pool = nn.MaxPool2d(pool_k)
            case 'avg':
                self.pool = nn.AvgPool2d(pool_k)
            case _:
                raise AssertionError("pool type not valid")

        self.conv1 = nn.Conv2d(input_channel, output_channel,
                               kernel_size=kernel_size, stride=stride, padding=padding)

        self.batch_norm = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout_p)

    def forward(self, X):
        x = self.conv1(X)
        x = self.batch_norm(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.dropout(x)
        return x
