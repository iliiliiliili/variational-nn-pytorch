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


def bbb_loss(
    sigma_0: float = 1
):

    log_likelihood_fn = torch.nn.CrossEntropyLoss()
    # log_likelihood_fn = torch.nn.NLLLoss()

    def calculate_single_kl(mu_q, sig_q, mu_p, sig_p):
        kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
        return kl

    def calculate_kl(net, batch_size):
        ws = net.hypertorso.ws
        bs = net.hypertorso.bs

        scales = [[torch.log(1 + torch.exp(w)) ** 2 for w in sub_ws] for sub_ws in ws]
        # biases = [[b ** 2 for b in sub_bs] for sub_bs in bs]

        kls = [[calculate_single_kl(0, sigma_0, b, s) for s, b in zip(sub_scales, sub_bs)] for sub_scales, sub_bs in zip(scales, bs)]
        kl = sum([sum(sub_kls) for sub_kls in kls])
        return kl


    def prior_kl(net, batch_size):

        ws = net.hypertorso.ws
        bs = net.hypertorso.bs

        scales = [[torch.log(1 + torch.exp(w)) ** 2 for w in sub_ws] for sub_ws in ws]
        biases = [[b ** 2 for b in sub_bs] for sub_bs in bs]

        sum_sq_scales = sum([sum([torch.sum(s ** 2) for s in sub_scales]) for sub_scales in scales])
        sum_log_scales = sum([sum([torch.sum(torch.log(s)) for s in sub_scales]) for sub_scales in scales])
        sum_sq_biases = sum([sum([torch.sum(s) for s in sub_biases]) for sub_biases in biases])
        count_biases = sum([sum([np.prod(s.shape) for s in sub_biases]) for sub_biases in biases])

        result = sum_sq_scales + sum_sq_biases / (sigma_0 ** 2) - count_biases - 2 * sum_log_scales
        result *= 0.5 / batch_size
        return result

    def single_loss(y, t, net, batch_size):
        prior_kl_loss = prior_kl(net, batch_size)
        # kl_loss = calculate_kl(net, batch_size)
        log_likelihood_loss = log_likelihood_fn(y, t)

        return prior_kl_loss + log_likelihood_loss

    return single_loss
