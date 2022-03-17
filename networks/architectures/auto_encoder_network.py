from networks.network import Network
from typing import Callable, List, Optional
import torch
from torch import nn


class AutoEncoderNetwork(Network):
    def __init__(self) -> None:
        super().__init__()

    def build(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        code_size: Optional[List[int]] = None,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.code_size = code_size

    def train_step(
        self,
        input,
        target,
        correct_count: Optional[
            Callable[[torch.Tensor, torch.Tensor], int]
        ] = None,
        clip_grad: Optional[float] = None,
    ):

        output = self(input)
        loss = self.loss_func(output, input)

        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

        self.optimizer.step()
        self.optimizer.zero_grad()

        if correct_count is None:
            return loss.item()
        else:
            return loss.item(), 0  # correct_count(output, target)

    def eval_step(
        self,
        input,
        target,
        correct_count: Optional[
            Callable[[torch.Tensor, torch.Tensor], int]
        ] = None,
    ):

        output = self(input)
        loss = self.loss_func(output, input)

        if correct_count is None:
            return loss.item()
        else:
            return loss.item(), 0  # correct_count(output, target)

    def generate(self, input=None, device=None):

        if input is None:

            if self.code_size is None:
                raise ValueError(
                    "Input is not defined and code_size to generate"
                    + " random input is unknown"
                )

            input = torch.rand(self.code_size, device=device)

        result = self.decoder(input)

        return result
