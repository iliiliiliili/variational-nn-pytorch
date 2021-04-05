from core import give, rename_dict
from typing import Any, Callable, Optional
from networks.variational import (
    VariationalBase,
)
import torch
from networks.auto_encoder_network import AutoEncoderNetwork
import fire  # type:ignore
import matplotlib.pyplot as plt
from params import (
    activation_params, activations, dataset_params, networks,
)
from torchvision import transforms


def display_as_images(
    values,
    get_image: Callable[[Any], Any],
    get_caption: Callable[[Any], str],
    line_size: int,
    save_path: Optional[str] = None,
    color_map: str = 'gray',
    axis: bool = False,
    size_multiplier: float = 2,
    show=True,
):
    count = len(values)

    lines = count // line_size + (
        0 if count % line_size == 0 else 1
    )

    fig = plt.figure(
        figsize=(
            int(line_size * size_multiplier),
            int(lines * size_multiplier)
        )
    )

    for i in range(count):
        subplot = fig.add_subplot(lines, line_size, i+1)

        if not axis:
            plt.axis('off')

        plt.imshow(get_image(values[i]), cmap=color_map)
        subplot.set_title(get_caption(values[i]))

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.close(fig)


def run_evaluation(net: AutoEncoderNetwork, dataset, device, images_count=40):

    print()

    net.eval()
    VariationalBase.GLOBAL_STD = 0

    images = []

    if dataset is None:

        for i in range(images_count):

            image = net.generate(device=device)

            images.append(image)

            print(
                "eval s["
                + str(i + 1)
                + "/"
                + str(images_count)
                + "]",
                end="\r",
            )
    else:
        for i, (data, target) in enumerate(dataset):

            data = data.to(device)

            image = net.generate(net.encoder(data), device=device)

            images.append(image)
            images.append(data)

            print(
                "eval s["
                + str(i + 1)
                + "/"
                + str(images_count)
                + "]",
                end="\r",
            )

            if i >= images_count:
                break

    return images


def create_train_validation_test(dataset_name: str):

    params = dataset_params[dataset_name]

    train_val = params["dataset"](
        params["path"],
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(params["mean"], params["std"]),
            ]
        ),
    )

    test = params["dataset"](
        params["path"],
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(params["mean"], params["std"]),
            ]
        ),
    )

    train, val = torch.utils.data.random_split(  # type: ignore
        train_val, [params["train_size"], params["validation_size"]]
    )

    return train, val, test


def generate(
    network_name,
    dataset_name,
    model_path=None,
    model_suffix="",
    device="cuda:0",
    split=None,
    image_shape=(28, 28),
    **kwargs,
):

    if "activation" in kwargs:

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

            if activation in current_activations[i + 1:]:
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

    if model_path is None:

        full_network_name = network_name

        if dataset_name not in network_name:
            full_network_name = dataset_name + "_" + full_network_name

        full_network_name += "" if model_suffix == "" else "_" + model_suffix

        model_path = "./models/" + full_network_name

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    net: AutoEncoderNetwork = networks[network_name](**kwargs)

    train, val, test = create_train_validation_test(dataset_name)

    current_dataset = None if split is None else {"train": train, "validation": val, "test": test}[split]

    images = run_evaluation(net, current_dataset, device)
    display_as_images(
        images,
        lambda img: img.view(image_shape).detach(),
        lambda img: '',
        6
    )


if __name__ == "__main__":

    fire.Fire()
