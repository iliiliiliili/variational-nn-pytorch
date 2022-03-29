from typing import Any, Optional, Tuple, Union
from torch import nn
import torch
import numpy as np

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


def negative_elbo_loss(
):

    noise_std = noise_scale * noise_std
    loss_fn = torch.nn.MSELoss()

    def noise_fn(x):
        x + torch.normal(torch.zeros_like(x), noise_std)

    def single_loss(y, t):
        loss_fn(y, noise_fn(t))

    return single_loss


def bbb_loss(
    sigma_0: float = 100
):

    log_likelihood_fn = torch.nn.CrossEntropyLoss()

    def prior_kl(net, batch_size):

        ws = net.hypertorso.ws
        bs = net.hypertorso.bs

        scales = [[torch.log(1 + torch.exp(w)) ** 2 for w in sub_ws] for sub_ws in ws]
        biases = [[b ** 2 for b in sub_bs] for sub_bs in bs]

        sum_sq_scales = sum([sum([torch.sum(s ** 2) for s in sub_scales]) for sub_scales in scales])
        sum_log_scales = sum([sum([torch.sum(torch.log(s)) for s in sub_scales]) for sub_scales in scales])
        sum_sq_biases = sum([sum([torch.sum(s * 2) for s in sub_biases]) for sub_biases in biases])
        count_biases = sum([sum([np.prod(s.shape) for s in sub_biases]) for sub_biases in biases])

        result = sum_sq_scales + sum_sq_biases / (sigma_0 ** 2) - count_biases - 2 * sum_log_scales
        result *= 0.5 / batch_size
        return result

    def single_loss(y, t, net, batch_size):
        prior_kl_loss = prior_kl(net, batch_size)
        log_likelihood_loss = log_likelihood_fn(y, t)

        return prior_kl_loss + log_likelihood_loss

    return single_loss
