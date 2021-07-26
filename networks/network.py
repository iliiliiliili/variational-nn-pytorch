from typing import Callable, List, Optional
import torch
from torch import nn
import os

from torch.functional import Tensor
from metrics import MeanStdMetric, AverageMetric


class Network(nn.Module):
    def __init__(self) -> None:
        self.uncertainty_value = None
        super().__init__()

    def prepare_train(
        self, optimizer, optimizer_params,
        loss_func=nn.CrossEntropyLoss(),
        uncertainty_loss_func=nn.MSELoss(),
    ):

        self.optimizer = optimizer(self.parameters(), **optimizer_params)
        self.loss_func = loss_func
        self.uncertainty_loss_func = uncertainty_loss_func

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
        loss = self.loss_func(output, target)

        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

        self.optimizer.step()
        self.optimizer.zero_grad()

        loss_dict = {
            "loss": loss.item(),
        }

        if correct_count is None:
            return loss_dict
        else:
            return loss_dict, correct_count(output, target)

    def train_step_uncertainty(
        self,
        input,
        target,
        correct_count: Optional[
            Callable[[torch.Tensor, torch.Tensor], int]
        ] = None,
        clip_grad: Optional[float] = None,
        monte_carlo_steps: int = 5,
        uncertainty_loss_weight: float = 0.2,
    ):

        mean_losses: List[Tensor] = []
        mean_std_outputs = MeanStdMetric()
        mean_uncertainty = AverageMetric()

        correctness = None

        for step in range(monte_carlo_steps):
            output = self(input)
            loss = self.loss_func(output, target)

            if correct_count is not None and correctness is None:
                correctness = correct_count(output, target)

            mean_losses.append(loss)
            mean_std_outputs.update(output)
            mean_uncertainty.update(
                self.uncertainty(method="uncertainty_layer")
            )

        monte_carlo_uncertainty = mean_std_outputs.get()[1]
        uncertainty_loss = self.uncertainty_loss_func(
            mean_uncertainty.get(), monte_carlo_uncertainty
        )

        mean_loss = sum(mean_losses) / monte_carlo_steps

        loss = (
            uncertainty_loss * uncertainty_loss_weight
            + mean_loss
        )

        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

        self.optimizer.step()
        self.optimizer.zero_grad()

        loss_dict = {
            "loss": loss.item(),
            "mean_loss": mean_loss.item(),
            "uncertainty_loss": uncertainty_loss.item(),
        }

        if correctness is None:
            return loss_dict
        else:
            return loss_dict, correctness

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

        loss_dict = {
            "loss": loss,
        }

        if correct_count is None:
            return loss_dict
        else:
            return loss_dict, correct_count(output, target)

    def save(self, save_path):

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        torch.save(self.state_dict(), save_path + "/model.pth")
        torch.save(self.optimizer.state_dict(), save_path + "/optimizer.pth")

    def load(self, load_path, device=None):
        self.load_state_dict(
            torch.load(load_path + "/model.pth", map_location=device)
        )

    def uncertainty(self, method="uncertainty_layer", params=None):

        if (
            method == "uncertainty_layer"
            and self.uncertainty_value is not None
        ):
            return self.uncertainty_value
        elif method == "monte-carlo":
            if "repeats" not in params:
                params["repeats"] = 10

            mean_std_metric = MeanStdMetric()

            for i in range(params["repeats"]):
                mean_std_metric.update(
                    self(params["input"]).detach().cpu().numpy()
                )
                print()

            return mean_std_metric.get()

        raise Exception("No such uncertainty method available for this model")
