from perturbed_evaluation import (
    evaluate_bar_occluded,
    evaluate_gaussian,
    evaluate_randomly_occluded,
    evaluate_randomly_swapped,
    evaluate_uniform,
)
from attacked_evaluation import evaluate_attacked
from metrics import AverageMetric
from networks.variational import VariationalBase
from typing import Optional
from core import give, rename_dict
import torch
from networks.network import Network
import os
import fire  # type:ignore
import json

from params import (
    create_network,
    dataset_params,
    loss_functions,
    loss_functions_that_use_network,
    loss_params,
    activations,
    activation_params,
    optimizer_params,
    optimizers,
)
import time
from tensorboardX import SummaryWriter


def create_train_validation_test(params):

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
        params["path"], train=True, download=True, transform=transform_train,
    )

    test = params["dataset"](
        params["path"], train=False, download=True, transform=transform_test,
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


def run_evaluation(net: Network, val, device, correct_count, batch, samples, eval_step=None):

    if eval_step is None:
        eval_step = net.eval_step

    print()

    net.eval()
    # VariationalBase.GLOBAL_STD = 0

    accuracy_metric = AverageMetric()
    accuracy_metric2 = None
    fps_metric = AverageMetric()
    fps_metric.skip(3)

    for i, (data, target) in enumerate(val):

        data = data.to(device)
        target = target.to(device)

        t = time.time()

        loss_dict, correct = eval_step(  # type: ignore
            data, target, correct_count=correct_count, samples=samples
        )

        t = time.time() - t
        fps_metric.update(1 / t)

        if isinstance(correct, list):
            if accuracy_metric2 is None:
                accuracy_metric2 = AverageMetric()

            accuracy_metric2.update(float(correct[1]) / batch)
            correct = correct[0]

        accuracy_metric.update(float(correct) / batch)

        print(
            "eval s["
            + str(i + 1)
            + "/"
            + str(len(val))
            + "]"
            + " loss="
            + ", ".join([str(k) + "=" + str(loss_dict[k]) for k in loss_dict.keys()])
            + " acc="
            + str(accuracy_metric.get())
            + (
                (" acc2=" + str(accuracy_metric2.get()))
                if accuracy_metric2 is not None
                else ""
            )
            + (" g_std=" + str(VariationalBase.GLOBAL_STD) if VariationalBase.GLOBAL_STD != 0 else "")
            + " fps="
            + str(fps_metric.get(-1)),
            end="\r",
        )

    print()

    net.train()
    return accuracy_metric.get(), fps_metric.get(-1)


def create_model_description(path, **kwargs):

    with open(path + "/training_parameters.json", "w") as f:
        json.dump(kwargs, f)


def load_training_parameters(path):

    with open(path) as f:
        params = json.load(f)

    return params


def evaluate(
    network_name=None,
    network_type=None,
    dataset_name=None,
    restore_training_parameters=True,
    use_best=True,
    batch=1,
    samples=1,
    model_path=None,
    model_suffix="",
    split="validation",
    device="cuda:0",
    evaluation_type="normal",
    attack_types=None,
    save=True,
    start_global_std: Optional[float] = None,
    end_global_std: Optional[float] = None,
    **kwargs,
):

    if model_path is None:

        full_network_name = network_name

        if dataset_name not in network_name:
            full_network_name = dataset_name + "_" + full_network_name

        full_network_name += "_" + network_type
        full_network_name += "" if model_suffix == "" else "_" + model_suffix

        model_path = "./models/" + full_network_name

    parameters = {}

    non_kwargs = [
        "network_name",
        "network_type",
        "dataset_name",
        "restore_training_parameters",
        "use_best",
        "batch",
        "samples",
        "model_path",
        "model_suffix",
        "split",
        "device",
        "evaluation_type",
        "attack_types",
        "save",
        "loss_sigma_0",
        "start_global_std",
        "end_global_std",
    ]

    given_parameters = {
        "network_name": network_name,
        "network_type": network_type,
        "dataset_name": dataset_name,
        "restore_training_parameters": restore_training_parameters,
        "use_best": use_best,
        "batch": batch,
        "samples": samples,
        "model_path": model_path,
        "model_suffix": model_suffix,
        "split": split,
        "device": device,
        "evaluation_type": evaluation_type,
        "attack_types": attack_types,
        "save": save,
        **kwargs,
    }

    if restore_training_parameters:
        training_parameters = load_training_parameters(
            model_path + "/training_parameters.json"
        )

        ignored_parameters = [
            "epochs",
            "save_steps",
            "validation_steps",
            "loss",
            "device",
            "save_best",
            # "start_global_std",
            # "end_global_std",
        ]

        for key, val in training_parameters.items():
            if (key not in ignored_parameters) and ("optimizer" not in key):
                parameters[key] = val

    for key, val in given_parameters.items():
        if (val is not None) or (key not in parameters):
            parameters[key] = val

    network_name = parameters["network_name"]
    network_type = parameters["network_type"]
    dataset_name = parameters["dataset_name"]
    use_best = parameters["use_best"]
    batch = parameters["batch"]
    samples = parameters["samples"]
    model_path = parameters["model_path"]
    model_suffix = parameters["model_suffix"]
    split = parameters["split"]
    device = parameters["device"]
    evaluation_type = parameters["evaluation_type"]
    attack_types = parameters["attack_types"]
    save = parameters["save"]
    start_global_std = parameters["start_global_std"]
    end_global_std = parameters["end_global_std"]

    if use_best:
        model_path += "/best"

    kwargs = {}
    for key, val in parameters.items():
        if key not in non_kwargs:
            kwargs[key] = val

    if "activation" in kwargs:
        kwargs = process_activation_kwargs(kwargs)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    net: Network = create_network(network_name, network_type)(**kwargs)

    net.load(model_path, device)
    net.to(device)

    result = None
    fps = None
    VariationalBase.GLOBAL_STD = end_global_std

    if evaluation_type == "normal":

        train, val, test = create_train_validation_test(dataset_params[dataset_name])

        current_dataset = {"train": train, "validation": val, "test": test}[split]
        current_dataset = torch.utils.data.DataLoader(  # type: ignore
            current_dataset, batch, shuffle=False, num_workers=4
        )

        result, fps = run_evaluation(net, current_dataset, device, correct_count, batch, samples)
    elif evaluation_type in [
        "gaussian",
        "uniform",
        "bar_occluded",
        "randomly_occluded",
        "randomly_swapped",
    ]:

        def evaluate_current(current_dataset_params):
            train, val, test = create_train_validation_test(current_dataset_params)
            current_dataset = {"train": train, "validation": val, "test": test,}[split]
            current_dataset = torch.utils.data.DataLoader(  # type: ignore
                current_dataset, batch, shuffle=False, num_workers=0
            )

            return run_evaluation(net, current_dataset, device, correct_count, batch, samples)

        result = {
            "gaussian": evaluate_gaussian,
            "uniform": evaluate_uniform,
            "bar_occluded": evaluate_bar_occluded,
            "randomly_occluded": evaluate_randomly_occluded,
            "randomly_swapped": evaluate_randomly_swapped,
        }[evaluation_type](evaluate_current, dataset_name, **kwargs,)
    elif evaluation_type in [
        "attacked",
    ]:

        class Normalize(torch.nn.Module):
            def __init__(self, mean, std):
                super(Normalize, self).__init__()
                self.channels = len(mean)
                self.register_buffer("mean", torch.Tensor(mean))
                self.register_buffer("std", torch.Tensor(std))

            def forward(self, input):
                # Broadcasting
                mean = self.mean.reshape(1, self.channels, 1, 1)
                std = self.std.reshape(1, self.channels, 1, 1)
                return (input - mean) / std

        def evaluate_current(current_dataset_params, attack):
            train, val, test = create_train_validation_test(current_dataset_params)
            current_dataset = {"train": train, "validation": val, "test": test,}[split]
            current_dataset = torch.utils.data.DataLoader(  # type: ignore
                current_dataset, batch, shuffle=False, num_workers=0
            )

            if "mean" in current_dataset_params:
                normalized_net = torch.nn.Sequential(
                    Normalize(
                        current_dataset_params["mean"], current_dataset_params["std"],
                    ),
                    net,
                    torch.nn.Softmax(-1),
                ).to(device)
            else:
                normalized_net = net

            def attack_step(
                input, target, correct_count=None, samples=1,
            ):
                nonlocal normalized_net
                nonlocal net

                def sampled_net(net, normalized_net, input, samples):
                    average_output = None
                    average_loss = None

                    for step in range(samples):
                        output = normalized_net(input)
                        loss = (
                            (net.loss_func(output, target, net, net.batch) if net.loss_uses_network else net.loss_func(output, target)).item()
                            if hasattr(net, "loss_func")
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
                    
                    return average_output, average_loss

                adv_images = attack(input, target)
                output, loss = sampled_net(net, normalized_net, adv_images, samples)
                loss_dict = {
                    "loss": loss,
                }

                if correct_count is None:
                    return loss_dict
                else:
                    return (
                        loss_dict,
                        [
                            correct_count(output, target),
                            correct_count(sampled_net(net, normalized_net, input, samples)[0], target),
                        ],
                    )

            return run_evaluation(
                normalized_net,
                current_dataset,
                device,
                correct_count,
                batch,
                samples,
                eval_step=attack_step,
            )

        result = evaluate_attacked(
            attack_types, net, evaluate_current, dataset_name, **kwargs,
        )
    else:
        raise ValueError("evaluation_type '" + evaluation_type + "' is unknown")

    print(
        'Evaluation on "' + evaluation_type + "_" + split + '" result: ' + str(result)
    )

    if not os.path.exists(model_path + "/results"):
        os.mkdir(model_path + "/results")

    if save:
        with open(
            model_path + "/results/eval_" + evaluation_type + "_" + split + ".txt", "w",
        ) as f:
            f.write(str(result) + "\n")

    return result


def run_evaluation_uncertainty(
    net: Network, val, device, correct_count, batch, eval_step=None, std_range=2,
):

    if eval_step is None:
        eval_step = net.eval_step

    print()

    net.eval()
    # VariationalBase.GLOBAL_STD = 0
    # VariationalBase.LOG_STDS = True

    correct_count = 0
    total_count = 0
    tp, tn, fp, fn = 0, 0, 0, 0

    for i, (data, target) in enumerate(val):

        data = data.to(device)
        target = target.to(device)

        output = net(data)
        uncertainty = net.uncertainty()
        softmax_output = torch.nn.functional.softmax(output, -1)

        monte_carlo_mean, monte_carlo_uncertainty = net.uncertainty(
            "monte-carlo", {"input": data}
        )

        monte_carlo_mean = torch.tensor(monte_carlo_mean)
        monte_carlo_uncertainty = torch.tensor(monte_carlo_uncertainty)

        mx, mxi = torch.max(output, axis=-1)
        uncertain_output = output + std_range * uncertainty
        uncertain_output[0, mxi] = mx - uncertainty[0, mxi]
        is_certain = torch.max(uncertain_output, axis=-1)[1] == mxi

        mx_mc, mxi_mc = torch.max(monte_carlo_mean, axis=-1)
        uncertain_output_mc = monte_carlo_mean + monte_carlo_uncertainty
        uncertain_output_mc[0, mxi_mc] = (
            mx_mc - std_range * monte_carlo_uncertainty[0, mxi_mc]
        )
        is_certain_mc = torch.max(uncertain_output_mc, axis=-1)[1] == mxi_mc

        is_correct = is_certain == is_certain_mc

        if is_correct:
            correct_count += 1

        if is_certain:
            if is_certain_mc:
                tp += 1
            else:
                fp += 1
        else:
            if is_certain_mc:
                fn += 1
            else:
                tn += 1

        total_count += 1

        # print(
        #     "target",
        #     target,
        #     "(correct output)"
        #     if correct_count(output, target) > 0
        #     else "(wrong output)",
        # )
        # print("is_certain", is_certain)
        # print("is_certain_mc", is_certain_mc)

        # print("output", output)
        # print("softmax_output", softmax_output)
        # print("uncertainty", uncertainty)
        # print()
        # print("monte_carlo_mean", monte_carlo_mean)
        # print("monte_carlo_uncertainty", monte_carlo_uncertainty)
        # print(
        #     "(u/m) / (mu/mm): ",
        #     (uncertainty / output).detach().numpy()
        #     / (monte_carlo_uncertainty / monte_carlo_mean),
        # )
        # print()
        print(
            "\rcorrect:",
            correct_count / total_count,
            "tp:",
            tp,
            "tn:",
            tn,
            "fp:",
            fp,
            "fn:",
            fn,
        )

    return correct_count / total_count


def evaluate_uncertainty(
    network_name=None,
    dataset_name=None,
    restore_training_parameters=True,
    use_best=True,
    batch=1,
    model_path=None,
    model_suffix="",
    split="validation",
    device="cuda:0",
    evaluation_type="monte-carlo-vs-single",
    save=True,
    **kwargs,
):

    if model_path is None:

        full_network_name = network_name

        if dataset_name not in network_name:
            full_network_name = dataset_name + "_" + full_network_name

        full_network_name += "" if model_suffix == "" else "_" + model_suffix

        model_path = "./models/" + full_network_name

    parameters = {}

    non_kwargs = [
        "network_name",
        "dataset_name",
        "restore_training_parameters",
        "use_best",
        "batch",
        "model_path",
        "model_suffix",
        "split",
        "device",
        "evaluation_type",
        "save",
    ]

    given_parameters = {
        "network_name": network_name,
        "dataset_name": dataset_name,
        "restore_training_parameters": restore_training_parameters,
        "use_best": use_best,
        "batch": batch,
        "model_path": model_path,
        "model_suffix": model_suffix,
        "split": split,
        "device": device,
        "evaluation_type": evaluation_type,
        "save": save,
        **kwargs,
    }

    if restore_training_parameters:
        training_parameters = load_training_parameters(
            model_path + "/training_parameters.json"
        )

        ignored_parameters = [
            "epochs",
            "save_steps",
            "validation_steps",
            "loss",
            "device",
            "save_best",
            "start_global_std",
            "end_global_std",
        ]

        for key, val in training_parameters.items():
            if (key not in ignored_parameters) and ("optimizer" not in key):
                parameters[key] = val

    for key, val in given_parameters.items():
        if val is not None:
            parameters[key] = val

    network_name = parameters["network_name"]
    dataset_name = parameters["dataset_name"]
    use_best = parameters["use_best"]
    batch = parameters["batch"]
    model_path = parameters["model_path"]
    model_suffix = parameters["model_suffix"]
    split = parameters["split"]
    device = parameters["device"]
    evaluation_type = parameters["evaluation_type"]
    save = parameters["save"]

    if use_best:
        model_path += "/best"

    kwargs = {}
    for key, val in parameters.items():
        if key not in non_kwargs:
            kwargs[key] = val

    if "activation" in kwargs:
        kwargs = process_activation_kwargs(kwargs)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    net: Network = networks[network_name](**kwargs)

    net.load(model_path, device)
    net.to(device)

    result = None
    fps = None

    if evaluation_type == "monte-carlo-vs-single":

        train, val, test = create_train_validation_test(dataset_params[dataset_name])

        current_dataset = {"train": train, "validation": val, "test": test}[split]
        current_dataset = torch.utils.data.DataLoader(  # type: ignore
            current_dataset, batch, shuffle=False, num_workers=4
        )

        result, run_evaluation_uncertainty(
            net, current_dataset, device, correct_count, batch
        )
    else:
        raise ValueError("evaluation_type '" + evaluation_type + "' is unknown")

    print(
        'Evaluation on "' + evaluation_type + "_" + split + '" result: ' + str(result)
    )

    if not os.path.exists(model_path + "/results"):
        os.mkdir(model_path + "/results")

    if save:
        with open(
            model_path + "/results/eval_" + evaluation_type + "_" + split + ".txt", "w",
        ) as f:
            f.write(str(result) + "\n")

    return result


def process_activation_kwargs(kwargs):

    current_activations = kwargs["activation"].split(" ")
    activation_functions = []

    for i, activation in enumerate(current_activations):

        activation_kwargs, kwargs = give(
            kwargs,  # type: ignore
            list(  # type: ignore
                map(lambda a: activation + "_" + a, activation_params[activation],)
            ),
        )

        if activation in current_activations[i + 1 :]:
            kwargs = {**kwargs, **activation_kwargs}

        func = activations[activation](
            **rename_dict(
                activation_kwargs, lambda name: name.replace(activation + "_", ""),
            )
        )

        activation_functions.append(func)

    if len(activation_functions) == 1:
        activation_functions = activation_functions[0]

    kwargs["activation"] = activation_functions

    if ("activation_mode" in kwargs) and (kwargs["activation_mode"] == "none"):
        kwargs["activation"] = activations["none"]()

    return kwargs


def process_optimizer_kwargs(optimizer_name, kwargs):

    optimizer_kwargs, kwargs = give(
        kwargs,
        list(  # type: ignore
            map(lambda a: "optimizer_" + a, optimizer_params[optimizer_name])
        ),
    )

    current_optimizer_params = rename_dict(
        optimizer_kwargs, lambda name: name.replace("optimizer_", ""),
    )

    current_optimizer_params

    return kwargs, current_optimizer_params


def process_loss_kwargs(loss_name, kwargs):

    loss_kwargs, kwargs = give(
        kwargs,
        list(  # type: ignore
            map(lambda a: "loss_" + a, loss_params[loss_name])
        ),
    )

    current_loss_params = rename_dict(
        loss_kwargs, lambda name: name.replace("loss_", ""),
    )

    current_loss_params

    return kwargs, current_loss_params


def train(
    network_name,
    network_type,
    dataset_name,
    batch,
    epochs,
    samples=1,
    model_path=None,
    model_suffix="",
    save_steps=-1,
    validation_steps=-1,
    optimizer=None,
    loss="cross_entropy",
    device="cuda:0",
    save_best=True,
    start_global_std: Optional[float] = None,
    end_global_std: Optional[float] = None,
    train_uncertainty: bool = False,
    monte_carlo_steps: int = 5,
    allow_retrain: bool = True,
    all_models_path = "./models",
    **kwargs,
):

    if model_path is None:

        full_network_name = network_name

        if dataset_name not in network_name:
            full_network_name = dataset_name + "_" + full_network_name

        full_network_name += "_" + network_type
        full_network_name += "" if model_suffix == "" else "_" + model_suffix

        model_path = all_models_path + "/" + full_network_name
    else:
        full_network_name = ""

    if os.path.exists(model_path) and (not allow_retrain) and (os.path.exists(model_path + "/best/model.pth")):
        print("Already exists", model_path)
        return

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(model_path + "/results"):
        os.mkdir(model_path + "/results")
    if not os.path.exists(model_path + "/best"):
        os.mkdir(model_path + "/best")
    if not os.path.exists(model_path + "/summary"):
        os.mkdir(model_path + "/summary")

    writer = SummaryWriter(model_path + "/summary")

    if optimizer is None:
        optimizer = "SGD"
        kwargs["optimizer_lr"] = 0.001 / (monte_carlo_steps if train_uncertainty else 1)
        kwargs["optimizer_momentum"] = 0.9

    save_kwargs = {
        "network_name": network_name,
        "network_type": network_type,
        "dataset_name": dataset_name,
        "batch": batch,
        "epochs": epochs,
        "model_path": model_path,
        "model_suffix": model_suffix,
        "save_steps": save_steps,
        "validation_steps": validation_steps,
        "optimizer": optimizer,
        "loss": loss,
        "device": device,
        "save_best": save_best,
        "start_global_std": start_global_std,
        "end_global_std": end_global_std,
        **kwargs,
    }

    def create_current_model_description(path):
        create_model_description(
            path,
            **save_kwargs
        )

    create_current_model_description(model_path)

    if "activation" in kwargs:
        kwargs = process_activation_kwargs(kwargs)

    kwargs, current_optimizer_params = process_optimizer_kwargs(optimizer, kwargs)

    kwargs, current_loss_params = process_loss_kwargs(loss, kwargs)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    net: Network = create_network(network_name, network_type)(**kwargs)

    train, val, _ = create_train_validation_test(dataset_params[dataset_name])

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
        loss_func=loss_functions[loss](**current_loss_params),
        loss_uses_network=loss in loss_functions_that_use_network,
        batch=batch,
    )
    net.to(device)

    steps_count = len(train) * epochs

    def run_train():
        net.train()

        current_step = 0

        for epoch in range(epochs):

            accuracy_metric = AverageMetric()

            for i, (data, target) in enumerate(train):

                if start_global_std is not None:
                    VariationalBase.GLOBAL_STD = start_global_std + (
                        current_step / steps_count
                    ) * (end_global_std - start_global_std)

                data = data.to(device)
                target = target.to(device)

                current_step += 1

                if train_uncertainty:
                    loss_dict, correct = net.train_step_uncertainty(
                        data, target, correct_count=correct_count
                    )
                else:
                    loss_dict, correct = net.train_step(
                        data, target, correct_count=correct_count, samples=samples,
                    )

                accuracy_metric.update(float(correct) / batch)

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
                    + ", ".join(
                        [str(k) + "=" + str(loss_dict[k]) for k in loss_dict.keys()]
                    )
                    + " acc="
                    + str(accuracy_metric.get())
                )

                for loss, value in loss_dict.items():
                    writer.add_scalar(loss, value, current_step)

                writer.add_scalar("acc", accuracy_metric.get(), current_step)
                writer.add_scalar("epoch", epoch + 1, current_step)

                if start_global_std is not None:
                    log += " g_std=" + str(VariationalBase.GLOBAL_STD)
                    writer.add_scalar(
                        "Global_STD", VariationalBase.GLOBAL_STD, current_step
                    )

                print("{:<80}".format(log), end="\n")

                if current_step % save_steps == 0:
                    net.save(model_path)

                if current_step % validation_steps == 0:
                    val_acc = run_evaluation(net, val, device, correct_count, batch, samples)

                    if validation_steps % len(train) == 0:
                        text = "epoch " + str(epoch + 1) + ": "
                    else:
                        text = "step " + str(current_step) + ": "

                    text += str(val_acc)

                    writer.add_scalar(
                        "val_acc",
                        (val_acc[0] if isinstance(val_acc, tuple) else val_acc),
                        current_step,
                    )

                    with open(
                        model_path + "/results/validation_batch_" + str(batch) + "samples_" + str(samples) + ".txt",
                        "a",
                    ) as f:
                        f.write(text + "\n")

                    best_description = get_best_description(
                        model_path + "/best/description.json"
                    )

                    is_should_save_best = False

                    if best_description is None:
                        is_should_save_best = True
                    else:
                        is_should_save_best = best_description["result"] * 1.001 < (
                            val_acc[0] if isinstance(val_acc, tuple) else val_acc
                        )

                    if is_should_save_best and save_best:
                        print(":::Saving Best:::")

                        net.save(model_path + "/best")

                        data = {
                            "epoch": epoch + 1,
                            "batch": batch,
                            "samples": samples,
                            "result": (
                                val_acc[0] if isinstance(val_acc, tuple) else val_acc
                            ),
                        }

                        with open(model_path + "/best/description.json", "w") as file:
                            json.dump(data, file)

                        create_current_model_description(model_path + "/best")

                        print(":::Saved Best:::")

    run_train()


if __name__ == "__main__":

    fire.Fire()
