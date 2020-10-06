from networks.variational import VariationalBase
from typing import Optional
from core import give, rename_dict
import torch
from torchvision import transforms
from networks.network import Network
import os
import fire  # type:ignore
import json

from params import (
    dataset_params,
    networks,
    loss_functions,
    activations,
    activation_params,
    optimizer_params,
    optimizers,
)


def create_train_validation_test(dataset_name: str):

    params = dataset_params[dataset_name]

    transform_train = (
        params["transform"]["train"]
        if "train" in params["transform"]
        else params["transform"]["all"]
    )
    transform_test = (
        params["transform"]["test"]
        if "test" in params["transform"]
        else params["transform"]["all"]
    )

    train_val = params["dataset"](
        params["path"],
        train=True,
        download=True,
        transform=transform_train,
    )

    test = params["dataset"](
        params["path"],
        train=False,
        download=True,
        transform=transform_test,
    )

    train, val = torch.utils.data.random_split(  # type: ignore
        train_val, [params["train_size"], params["validation_size"]]
    )

    return train, val, test


def get_best_description(path):
    if os.path.exists(path):
        with open(path) as file:
            data = json.load(file)
            return data
    else:
        return None


def correct_count(output, target):
    labels = output.data.max(1, keepdim=True)[1]
    return labels.eq(target.data.view_as(labels)).sum()


def run_evaluation(net: Network, val, device, correct_count, batch):

    print()

    net.eval()
    VariationalBase.GLOBAL_STD = 0

    total_correct = 0
    total_elements = 0

    for i, (data, target) in enumerate(val):

        data = data.to(device)
        target = target.to(device)

        loss, correct = net.eval_step(
            data, target, correct_count=correct_count
        )

        total_correct += correct
        total_elements += batch

        print(
            "eval s["
            + str(i + 1)
            + "/"
            + str(len(val))
            + "]"
            + " loss="
            + str(loss)
            + " acc="
            + str(float(total_correct) / total_elements),
            end="\r",
        )

    print()

    net.train()
    return float(total_correct) / total_elements


def evaluate(
    network_name,
    dataset_name,
    batch=1,
    model_path=None,
    model_suffix="",
    split="validation",
    device="cuda:0",
    **kwargs,
):

    if model_path is None:

        full_network_name = network_name

        if dataset_name not in network_name:
            full_network_name = dataset_name + "_" + full_network_name

        full_network_name += "" if model_suffix == "" else "_" + model_suffix

        model_path = "./models/" + full_network_name

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    net: Network = networks[network_name](**kwargs)

    train, val, test = create_train_validation_test(dataset_name)

    current_dataset = {"train": train, "validation": val, "test": test}[split]

    result = run_evaluation(net, current_dataset, device, correct_count, batch)

    print('Evaluation on "' + split + '" result: ' + str(result))

    with open(model_path + "/results/eval_" + split + ".txt", "w") as f:
        f.write(str(result) + "\n")


def process_activation_kwargs(kwargs):

    current_activations = kwargs["activation"].split(" ")
    activation_functions = []

    for i, activation in enumerate(current_activations):

        activation_kwargs, kwargs = give(
            kwargs,
            list(
                map(
                    lambda a: activation + "_" + a,
                    activation_params[activation],
                )
            ),
        )

        if activation in current_activations[i + 1 :]:
            kwargs = {**kwargs, **activation_kwargs}

        func = activations[activation](
            **rename_dict(
                activation_kwargs,
                lambda name: name.replace(activation + "_", ""),
            )
        )

        activation_functions.append(func)

    if len(activation_functions) == 1:
        activation_functions = activation_functions[0]

    kwargs["activation"] = activation_functions

    return kwargs


def process_optimizer_kwargs(optimizer_name, kwargs):

    optimizer_kwargs, kwargs = give(
        kwargs,
        list(
            map(lambda a: "optimizer_" + a, optimizer_params[optimizer_name],)
        ),
    )

    current_optimizer_params = rename_dict(
        optimizer_kwargs, lambda name: name.replace("optimizer_", ""),
    )

    current_optimizer_params

    return kwargs, current_optimizer_params


def train(
    network_name,
    dataset_name,
    batch,
    epochs,
    model_path=None,
    model_suffix="",
    save_steps=-1,
    validation_steps=-1,
    optimizer=None,
    loss_function_name="cross_entropy",
    device="cuda:0",
    save_best=True,
    start_global_std: Optional[float] = None,
    end_global_std: Optional[float] = None,
    **kwargs,
):

    if "activation" in kwargs:
        kwargs = process_activation_kwargs(kwargs)

    if optimizer is None:
        optimizer = "SGD"
        kwargs["optimizer_lr"] = 0.001
        kwargs["optimizer_momentum"] = 0.9

    kwargs, current_optimizer_params = process_optimizer_kwargs(
        optimizer, kwargs
    )

    if model_path is None:

        full_network_name = network_name

        if dataset_name not in network_name:
            full_network_name = dataset_name + "_" + full_network_name

        full_network_name += "" if model_suffix == "" else "_" + model_suffix

        model_path = "./models/" + full_network_name
    else:
        full_network_name = ""

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(model_path + "/results"):
        os.mkdir(model_path + "/results")
    if not os.path.exists(model_path + "/best"):
        os.mkdir(model_path + "/best")

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    net: Network = networks[network_name](**kwargs)

    train, val, _ = create_train_validation_test(dataset_name)

    train = torch.utils.data.DataLoader(  # type: ignore
        train, batch, shuffle=True, num_workers=4
    )

    val = torch.utils.data.DataLoader(  # type: ignore
        val, batch, shuffle=False, num_workers=4
    )

    if save_steps < 0:
        save_steps = -save_steps * len(train)

    if validation_steps < 0:
        validation_steps = -validation_steps * len(train)

    net.prepare_train(
        optimizer=optimizers[optimizer],
        optimizer_params=current_optimizer_params,
        loss_func=loss_functions[loss_function_name],
    )
    net.to(device)

    steps_count = len(train) * epochs

    def run_train():
        net.train()

        current_step = 0

        for epoch in range(epochs):

            total_correct = 0
            total_elements = 0

            for i, (data, target) in enumerate(train):

                if start_global_std is not None:
                    VariationalBase.GLOBAL_STD = start_global_std + (
                        current_step / steps_count
                    ) * (end_global_std - start_global_std)

                data = data.to(device)
                target = target.to(device)

                current_step += 1
                loss, correct = net.train_step(
                    data, target, correct_count=correct_count
                )

                total_correct += correct
                total_elements += batch

                log = (
                    full_network_name
                    + " e["
                    + str(epoch + 1)
                    + "/"
                    + str(epochs)
                    + "]"
                    + " s["
                    + str(i + 1)
                    + "/"
                    + str(len(train))
                    + "]"
                    + " loss="
                    + str(loss)
                    + " acc="
                    + str(float(total_correct) / total_elements)
                )

                if start_global_std is not None:
                    log += " g_std=" + str(VariationalBase.GLOBAL_STD)

                print("{:<80}".format(log), end="\n")

                if current_step % save_steps == 0:
                    net.save(model_path)

                if current_step % validation_steps == 0:
                    val_acc = run_evaluation(
                        net, val, device, correct_count, batch
                    )

                    if validation_steps % len(train) == 0:
                        text = "epoch " + str(epoch + 1) + ": "
                    else:
                        text = "step " + str(current_step) + ": "

                    text += str(val_acc)

                    with open(
                        model_path
                        + "/results/validation_batch_"
                        + str(batch)
                        + ".txt",
                        "a",
                    ) as f:
                        f.write(text)

                    best_description = get_best_description(
                        model_path + "/best/description.json"
                    )

                    is_should_save_best = False

                    if best_description is None:
                        is_should_save_best = True
                    else:
                        is_should_save_best = (
                            best_description["result"] * 1.001 < val_acc
                        )

                    if is_should_save_best and save_best:
                        print(":::Saving Best:::")

                        net.save(model_path + "/best")

                        data = {
                            "epoch": epoch + 1,
                            "batch": batch,
                            "result": val_acc,
                        }

                        with open(
                            model_path + "/best/description.json", "w"
                        ) as file:
                            json.dump(data, file)

                        print(":::Saved Best:::")

    run_train()


if __name__ == "__main__":

    fire.Fire()
