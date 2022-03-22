from typing import Any, Optional, Tuple, Union
from torch import nn
import torch

from networks.network import Network

def gaussian_regression_loss(
    noise_std: float = 0.1,
    noise_scale: float = 1,
):
    """Add a matching Gaussian noise to the target y."""

    noise_std = noise_scale * noise_std
    loss_fn = torch.nn.MSELoss()

    def noise_fn(x):
        x + torch.normal(torch.zeros_like(x), noise_std)

    def single_loss(y, t):
        loss_fn(y, noise_fn(t))

    return single_loss
