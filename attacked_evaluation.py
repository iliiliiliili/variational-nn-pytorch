import torchattacks
import torch
from params import (
    SEED,
    attacked_dataset_params,
)


def evaluate_attacked(attack_types, model, evaluate, dataset_name, **kwargs):

    torch.manual_seed(SEED)

    attacks = {
        "FGSM": lambda model: torchattacks.FGSM(model, eps=8 / 255),
        "BIM": lambda model: torchattacks.BIM(
            model, eps=8 / 255, alpha=2 / 255, steps=7
        ),
        "CW": lambda model: torchattacks.CW(
            model, c=10, kappa=0, steps=1000, lr=0.01
        ),
        "RFGSM": lambda model: torchattacks.RFGSM(
            model, eps=8 / 255, alpha=4 / 255, steps=1
        ),
        "PGD": lambda model: torchattacks.PGD(
            model, eps=8 / 255, alpha=2 / 255, steps=7
        ),
        "FFGSM": lambda model: torchattacks.FFGSM(
            model, eps=8 / 255, alpha=12 / 255
        ),
        "TPGD": lambda model: torchattacks.TPGD(
            model, eps=8 / 255, alpha=2 / 255, steps=7
        ),
        "MIFGSM": lambda model: torchattacks.MIFGSM(
            model, eps=8 / 255, decay=1.0, steps=5
        ),
        "APGD": lambda model: torchattacks.APGD(model, eps=8 / 255, steps=10),
        "FAB": lambda model: torchattacks.FAB(model, eps=8 / 255),
        "Square": lambda model: torchattacks.Square(model, eps=8 / 255),
        "PGDDLR": lambda model: torchattacks.PGDDLR(
            model, eps=8 / 255, alpha=2 / 255, steps=7
        ),
        "DeepFool": lambda model: torchattacks.DeepFool(
            model, steps=50, overshoot=0.02
        ),
        "OnePixel": lambda model: torchattacks.OnePixel(
            model, pixels=1, steps=75, popsize=400
        ),
        "SparseFool": lambda model: torchattacks.SparseFool(
            model, steps=20, lam=3, overshoot=0.02
        ),
    }

    results = []

    for attack_type in attack_types:
        attack = attacks[attack_type](model)
        result, fps = evaluate(attacked_dataset_params[dataset_name], attack)
        results.append(
            "attack: "
            + str(attack_type)
            + " result: "
            + str(result)
            + " fps: "
            + str(fps)
        )

    return "\n".join(results)
