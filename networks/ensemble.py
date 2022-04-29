from torch import nn
import torch

from networks.network import Network


class EnsembleNetwork(Network):
    def __init__(self, networks, priors, prior_scale) -> None:
        super().__init__()
        self.networks = nn.ModuleList(networks)
        self.priors = nn.ModuleList(priors)
        self.prior_scale = prior_scale

    def forward(self, x):

        def output_with_prior(network, prior):
            result = network(x)
            if self.prior_scale > 0:
                return result + prior(x) * self.prior_scale

            return result

        result = torch.mean(torch.stack([output_with_prior(network, prior) for network, prior in zip(self.networks, self.priors)]), dim=0)
        return result


def create_ensemble(network_creator, num_ensemble=10, prior_scale=1, **kwargs):
    networks = [network_creator(**kwargs) for _ in range(num_ensemble)]
    if prior_scale > 0: 
        priors = [network_creator(**kwargs) for _ in range(num_ensemble)]
        [p.requires_grad_(False) for p in priors]
    else:
        priors = [nn.Identity() for _ in range(num_ensemble)]

    result = EnsembleNetwork(networks, priors, prior_scale)

    return result