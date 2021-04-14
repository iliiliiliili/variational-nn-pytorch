import torch
from torchvision import datasets, transforms
from network import Network

# import fire

from base_vnn import BaseVNN
from cifar10_base_vnn import Cifar10BaseVNN
from resnet_vnn import (
    ResNetVNN18,
    ResNetVNN34,
    ResNetVNN50,
    ResNetVNN101,
    ResNetVNN152,
)

dataset_params = {
    "mnist": {
        "mean": (0.1307,),
        "std": (0.3081,),
        "dataset": datasets.MNIST,
        "path": "./datasets/",
        "train_size": 50000,
        "validation_size": 10000,
    },
    "cifar10": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "dataset": datasets.CIFAR10,
        "path": "./datasets/",
        "train_size": 40000,
        "validation_size": 10000,
    },
}


networks = {
    "base_vnn": BaseVNN,
    "cifar10_base_vnn": Cifar10BaseVNN,
    "resnet_vnn_18": ResNetVNN18,
    "resnet_vnn_34": ResNetVNN34,
    "resnet_vnn_50": ResNetVNN50,
    "resnet_vnn_101": ResNetVNN101,
    "resnet_vnn_152": ResNetVNN152,
}


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

    train, val = torch.utils.data.random_split(
        train_val, [params["train_size"], params["validation_size"]]
    )

    return train, val, test


def train(
    net: Network,
    batch,
    dataset_name,
    epochs,
    save_path,
    device,
    save_steps=-1,
    validation_steps=-1,
    **kwargs
):

    train, val, test = create_train_validation_test(dataset_name)

    train = torch.utils.data.DataLoader(
        train, batch, shuffle=True, num_workers=1
    )

    val = torch.utils.data.DataLoader(val, batch, shuffle=False, num_workers=1)

    test = torch.utils.data.DataLoader(
        test, batch, shuffle=False, num_workers=1
    )

    if save_steps < 0:
        save_steps = -save_steps * len(train)

    if validation_steps < 0:
        validation_steps = -validation_steps * len(train)

    net.prepare_train(**kwargs)
    net.to(device)

    def correct_count(output, target):
        labels = output.data.max(1, keepdim=True)[1]
        return labels.eq(target.data.view_as(labels)).sum()

    def run_validation():

        print()

        net.eval()

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

    def run_train():
        net.train()

        current_step = 0

        for epoch in range(epochs):

            total_correct = 0
            total_elements = 0

            for i, (data, target) in enumerate(train):

                data = data.to(device)
                target = target.to(device)

                current_step += 1
                loss, correct = net.train_step(
                    data, target, correct_count=correct_count
                )

                total_correct += correct
                total_elements += batch

                print(
                    "{:<80}".format(
                        "e["
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
                    ),
                    end="\n",
                )

                if current_step % save_steps == 0:
                    net.save(save_path)

                if current_step % validation_steps == 0:
                    run_validation()

    run_train()


if __name__ == "__main__":

    # fire.Fire()

    # net = BaseVNN()
    # net = Cifar10BaseVNN(use_batch_norm=True, bias=False)
    net = ResNetVNN50()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train(
        net,
        100,
        "cifar10",
        100,
        "models/vnn/base-cifar10",
        device,
        learning_rate=0.001,
    )
