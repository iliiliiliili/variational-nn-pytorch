from typing import Callable, Optional
import torch
from torch import nn


class Network(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def prepare_train(
        self,
        learning_rate=0.0001,
        momentum=0.9,
        loss_func=nn.CrossEntropyLoss()
    ):
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=learning_rate, momentum=momentum
        )

        self.loss_func = loss_func

    def train_step(
        self, input, target,
        correct_count: Optional[
            Callable[[torch.Tensor, torch.Tensor], int]
        ] = None,
        clip_grad: Optional[float] = None
    ):

        output = self(input)
        loss = self.loss_func(output, target)

        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

        self.optimizer.step()
        self.optimizer.zero_grad()

        if correct_count is None:
            return loss.item()
        else:
            return loss.item(), correct_count(output, target)

    def eval_step(
        self, input, target,
        correct_count: Optional[
            Callable[[torch.Tensor, torch.Tensor], int]
        ] = None,
    ):

        output = self(input)
        loss = self.loss_func(output, target)

        if correct_count is None:
            return loss.item()
        else:
            return loss.item(), correct_count(output, target)

    def save(self, save_path):
        torch.save(self.state_dict(), save_path + '/model.pth')
        torch.save(self.optimizer.state_dict(), save_path + '/optimizer.pth')
