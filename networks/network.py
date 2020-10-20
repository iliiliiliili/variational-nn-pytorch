from typing import Callable, Optional
import torch
from torch import nn
import os


class Network(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def prepare_train(
        self,
        optimizer,
        optimizer_params,
        loss_func=nn.CrossEntropyLoss()
    ):

        self.optimizer = optimizer(self.parameters(), **optimizer_params)
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
        self,
        input,
        target,
        correct_count: Optional[
            Callable[[torch.Tensor, torch.Tensor], int]
        ] = None,
    ):

        output = self(input)
        loss = (
            self.loss_func(output, target).item()
            if hasattr(self, "loss_func")
            else None
        )

        if correct_count is None:
            return loss
        else:
            return loss, correct_count(output, target)

    def save(self, save_path):

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        torch.save(self.state_dict(), save_path + "/model.pth")
        torch.save(self.optimizer.state_dict(), save_path + "/optimizer.pth")

    def load(self, load_path, device=None):
        self.load_state_dict(
            torch.load(load_path + "/model.pth", map_location=device)
        )
