import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import DenseLayer
from .layers import PositionalEncoding


class MLPLight(nn.Module):
    def __init__(self, D=8, W=256, embedding_res=10, out_channels=3):
        super(MLPLight, self).__init__()
        self.D = D
        self.W = W
        self.out_channels = out_channels
        self.embedder = PositionalEncoding(embedding_res, 3, True)
        self.input_ch = self.embedder.out_dim
        self.linears = nn.ModuleList(
            [DenseLayer(self.input_ch, W, activation="relu")]
            + [DenseLayer(W, W, activation="relu") for _ in range(D-1)])
        self.output = DenseLayer(W, out_channels, activation="linear")

        gamma = torch.tensor([1.0])
        self.gamma = nn.parameter.Parameter(gamma)

    def forward(self, input):
        x = self.embedder(input)
        h = x
        for layer in self.linears:
            h = layer(h)
            h = F.relu(h)
        light = self.output(h)
        return light

