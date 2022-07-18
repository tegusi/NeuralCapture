import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, L, in_dim=3, identity=True, gate='None'):
        super(PositionalEncoding, self).__init__()
        self.L = L
        self.identity = identity
        self.in_dim = in_dim
        self.gate_func = getattr(self, f"_{gate}_gate")
        self.out_dim = (2*L+1)*in_dim if identity else 2*L*in_dim

    def forward(self, x, alpha=None):
        if alpha is None:
            alpha = torch.zeros_like(x).detach()
        gate = self.gate_func(alpha)

        output = []
        if self.identity:
            output.append(x)
        for i in range(0, self.L):
            freq = np.power(2., i)
            output.append(torch.sin(x*freq))
            output.append(torch.cos(x*freq))
        output = torch.cat(output, dim=-1) * gate
        return output

    # gate function
    def _None_gate(self, alpha):
        return 1.0

    def _Hann_gate(self, alpha):
        r"""Hann filtering function in 
        Nerfies: Deformable Neural Radiance Fields

        https://arxiv.org/pdf/2011.12948.pdf
        """
        output = []
        if self.identity:
            output.append(torch.ones_like(alpha))
        for i in range(0, self.L):
            t = alpha * self.L
            w = 0.5*(1 - torch.cos(np.pi*torch.clip(t - i, 0., 1.)))
            output = output + [w] * 2
        output = torch.cat(output, dim=-1)
        return output

    def _IPE_gate(self, sigma2):
        r"""IPE filtering function as proposed in 

        Mip-NeRF: A Multiscale Representation for 
        Anti-Aliasing Neural Radiance Fields

        https://arxiv.org/pdf/2103.13415.pdf
        """
        output = []
        if self.identity:
            output.append(torch.ones_like(sigma2))
        for i in range(0, self.L):
            freq = np.power(4., i)
            w = torch.exp(-0.5*freq*sigma2)
            output = output + [w]*2
        output = torch.cat(output, dim=-1)
        return output
