import torchattacks
import torch
from params import (
    SEED,
    attacked_dataset_params,
)


def evaluate_attacked(attack_type, model, evaluate, dataset_name, **kwargs):

    torch.manual_seed(SEED)

    attack = {
        "PGD": lambda model: torchattacks.PGD(
            model, eps=8 / 255, alpha=2 / 255, steps=4
        ),
    }[attack_type](model)

    result = evaluate(attacked_dataset_params[dataset_name], attack)

    return result
