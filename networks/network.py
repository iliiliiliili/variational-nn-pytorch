from typing import Callable, List, Optional
import torch
from torch import nn
import os

from torch.functional import Tensor
from metrics import MeanStdMetric, AverageMetric


class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def prepare_train(
        self,
        optimizer,
        optimizer_params,
        loss_func=nn.CrossEntropyLoss(),
        loss_uses_network=False,
        monte_carlo_loss_func=nn.MSELoss(),
        batch=1,
    ):

        self.optimizer = optimizer(self.parameters(), **optimizer_params)
        self.loss_func = loss_func
        self.loss_uses_network = loss_uses_network
        self.monte_carlo_loss_func = monte_carlo_loss_func
        self.batch = batch

    def train_step(
        self,
        input,
        target,
        correct_count: Optional[
            Callable[[torch.Tensor, torch.Tensor], int]
        ] = None,
        clip_grad: Optional[float] = None,
        samples=1,
    ):

        average_loss = 0
        average_output = 0

        for step in range(samples):
            output = self(input)
            loss = self.loss_func(output, target, self, self.batch) if self.loss_uses_network else self.loss_func(output, target)

            average_loss += loss
            average_output += output
        
        average_loss /= samples
        average_output /= samples

        average_loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

        self.optimizer.step()
        self.optimizer.zero_grad()

        loss_dict = {
            "loss": average_loss.item(),
        }

        if correct_count is None:
            return loss_dict
        else:
            return loss_dict, correct_count(average_output, target)

    def train_step_uncertainty(
        self,
        input,
        target,
        correct_count: Optional[
            Callable[[torch.Tensor, torch.Tensor], int]
        ] = None,
        clip_grad: Optional[float] = None,
        monte_carlo_steps: int = 5,
        uncertainty_monte_carlo_loss_weight: float = 0.4,
        # mean_monte_carlo_loss_weight: float = 0.2,
        eps: float = 1e-5,
    ):

        mean_losses: List[Tensor] = []
        mean_std_outputs_metric = MeanStdMetric()
        mean_uncertainty_metric = AverageMetric()

        correctness = None

        for step in range(monte_carlo_steps):
            output = self(input)
            loss = self.loss_func(output, target)

            if correct_count is not None and correctness is None:
                correctness = correct_count(output, target)

            mean_losses.append(loss)
            mean_std_outputs_metric.update(output)
            mean_uncertainty_metric.update(
                self.uncertainty(method="uncertainty_layer")  # / (output + eps)
            )

        (
            monte_carlo_mean,
            monte_carlo_uncertainty,
        ) = mean_std_outputs_metric.get()
        uncertainty_monte_carlo_loss = self.monte_carlo_loss_func(
            mean_uncertainty_metric.get(),
            monte_carlo_uncertainty  # / (monte_carlo_mean + eps),
        )
        # mean_monte_carlo_loss = self.monte_carlo_loss_func(
        #     mean_uncertainty_metric.get(), monte_carlo_uncertainty
        # )

        mean_loss = sum(mean_losses) / monte_carlo_steps

        loss = (
            uncertainty_monte_carlo_loss * uncertainty_monte_carlo_loss_weight
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
            "uncertainty_monte_carlo_loss": uncertainty_monte_carlo_loss.item(),
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
        samples=1,
    ):

        average_output = None
        average_loss = None

        for step in range(samples):
            output = self(input)
            loss = (
                (self.loss_func(output, target, self, self.batch) if self.loss_uses_network else self.loss_func(output, target)).item()
                if hasattr(self, "loss_func")
                else None
            )

            if average_output is None:
                average_output = output
            else:
                average_output += output
            if average_loss is None:
                average_loss = loss
            else:
                average_loss += loss
        
        average_output /= samples
        
        if average_loss is not None:
            average_loss /= samples

        loss_dict = {
            "loss": average_loss,
        }

        if correct_count is None:
            return loss_dict
        else:
            return loss_dict, correct_count(average_output, target)

    def save(self, save_path):

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        torch.save(self.state_dict(), save_path + "/model.pth")
        torch.save(self.optimizer.state_dict(), save_path + "/optimizer.pth")

    def load(self, load_path, device=None):
        self.load_state_dict(
            torch.load(load_path + "/model.pth", map_location=device)
        )

    def uncertainty(self, method="monte-carlo", params=None):

        if method == "monte-carlo":
            if "repeats" not in params:
                params["repeats"] = 10

            mean_std_metric = MeanStdMetric()

            for i in range(params["repeats"]):
                mean_std_metric.update(
                    self(params["input"]).detach().cpu().numpy()
                )

            return mean_std_metric.get()

        raise Exception("No such uncertainty method available for this model")
