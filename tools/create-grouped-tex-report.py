import sys
import os
import json

from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    facet_grid,
    facet_wrap,
    scale_y_continuous,
    geom_hline,
    position_dodge,
    geom_errorbar,
    theme,
    element_text,
)
from plotnine.data import economics
from pandas import Categorical, DataFrame
from plotnine.scales.limits import ylim
from plotnine.scales.scale_xy import scale_x_discrete
from glob import glob
import re


models_root = "./models" if len(sys.argv) < 2 else sys.argv[1]
model_folders = [str(f.path) for f in os.scandir(models_root) if f.is_dir()]
output_file = (
    "tools/tex-reports/grouped-report.tex" if len(sys.argv) < 3 else sys.argv[2]
)
template_file = (
    "tools/tex-report-template.tex" if len(sys.argv) < 4 else sys.argv[3]
)
top_k = (
    2 if len(sys.argv) < 5 else int(sys.argv[4])
)

captions = "    Dataset & Model & Type & Activation & Optimizer & BN/DO & Accuracy\\\\ [0.5ex] \n        \\hline"
mode = "|c c c c c c c|"

with open(template_file, "r") as f:
    template = f.read()


def get_model_group(model_name):

    groups = [
        "densenet1",
        "densenet2",
        "densenet",
        "resnet",
        "vgg",
        "mini2 base",
        "mini base",
        "base",
        "mlp",
        "double linear",
    ]

    for g in groups:
        if g in model_name:
            return g

    return model_name


def parse_model_params(path):
    with open(path + "/training_parameters.json", "r") as f:
        params = json.load(f)

        network_type = params["network_type"]
        regularization_type = ""
        activation_mode = ""

        if "densenet" in params["network_name"]:
            regularization_type += "BN E"

        # if "dropout" in params["network_name"]:
        #     network_type = "Normal"
        #     regularization_type += "DO"

        #     if "dropout_type" in params:
        #         dropout_type = params["dropout_type"]
        #         regularization_type += " " + dropout_type

        elif "vnn" in network_type:

            if "global_std_mode" in params:
                network_type = {
                    "none": "vnn",
                    "replace": "vnn G",
                    "multiply": "vnn D",
                }[params["global_std_mode"]]

        if "batch_norm_mode" in params:
            batch_norm_mode = (
                params["batch_norm_mode"]
                .replace("+", "")
                .replace("mean", "M")
                .replace("std", "S")
                .replace("end", "E")
            )
            regularization_type += "BN " + batch_norm_mode

        if "activation_mode" in params:
            activation_mode = params["activation_mode"]

        activation_mode = (
            activation_mode
            .replace("+", " ")
            .replace("mean", "M")
            .replace("std", "S")
            .replace("end", "E")
        )

        network_name = (
            params["network_name"]
            .replace("_classic", "")
            .replace("_vnn", "")
            .replace("_dropout", "")
            .replace("mnist_", "")
            .replace("cifar10_", "")
        )

        activation = (
            params["activation"]
            .replace("+", " ")
            .replace("leacky_relu", "LRl")
            .replace("relu", "Rl")
            .replace("relu6", "Rl6")
            .replace("sigmoid", "Sg")
            .replace("tanh", "Th")
            .replace("noact", "")
        )

        if activation_mode != "":
            activations = activation.split(" ")
            activation_modes = activation_mode.split(" ")

            activation = []

            for a, m in zip(activations, activation_modes):
                activation.append(m + ":" + a)

            activation = " ".join(activation)

        result = [
            params["dataset_name"].replace("_", " "),
            network_name.replace("_", " "),
            network_type.replace("_", " "),
            activation.replace("_", " "),
            params["optimizer"].replace("_", " "),
            regularization_type.replace("_", " "),
        ]

        return result


elements = []
type_elemets = {}

for path in model_folders:

    print(path, end="\r")

    if os.path.exists(path + "/best/description.json") and os.path.exists(
        path + "/training_parameters.json"
    ):
        with open(path + "/best/description.json", "r") as f:
            try:
                best = json.load(f)

            except Exception:
                continue

            print()

            element = [
                *parse_model_params(path),
                "%.3f" % best["result"],
            ]

            type = get_model_group(element[1]) + element[2]

            if type not in type_elemets:
                type_elemets[type] = []

            type_elemets[type].append(element)

for type, local_elements in type_elemets.items():
    local_elements.sort(key=lambda x: (x[0], get_model_group(x[1]), x[-1]), reverse=True)
    
    if len(local_elements) > top_k:
        local_elements = local_elements[:top_k]
    
    elements = [*elements, *local_elements]

print()
print(len(elements))

elements.sort(key=lambda x: (x[0], get_model_group(x[1]), x[-1]), reverse=True)

table = [captions]

last_dataset = None
last_model = None

elements_in_a_group = 0

for element in elements:

    print(element)

    if element[0] != last_dataset:
        table.append("\\hline\\hline")
        last_dataset = element[0]
        last_model = get_model_group(element[1])
        elements_in_a_group = 0
    elif get_model_group(element[1]) != last_model:
        table.append("\\hline")
        last_model = get_model_group(element[1])
        elements_in_a_group = 0

    table.append(" & ".join(element) + " \\\\")
    elements_in_a_group += 1

table[-1] += " [1ex]"

table = "\n        ".join(table)

template = template.replace("<MODE>", mode)
template = template.replace("<TABLE>", table)

with open(output_file, "w") as f:
    f.write(template)

def plot(elements, output_file_name, exclude_methods=["classic"]):

    frame = {
        "Dataset": [],
        "Architecture": [],
        "Method": [],
        "Accuracy": [],
    }

    architecture_to_name = {
        "resnet 18": "Resnet 18",
        "mini2 base": "Micro Base",
        "mini base": "Mini Base",
        "base": "Base",
        "mlp": "MLP",
    }

    dataset_to_name = {
        "cifar10 n2": "CIFAR-10",
        "mnist": "MNIST",
    }

    for element in elements:
        if element[2] not in exclude_methods:
            frame["Dataset"].append(dataset_to_name[element[0]])
            frame["Architecture"].append(architecture_to_name[element[1]])
            frame["Method"].append(element[2])
            frame["Accuracy"].append(float(element[-1]))

    frame["Method"] = Categorical(frame["Method"], ["bbb", "dropout", "vnn", "ensemble", "hypermodel"])
    frame["Dataset"] = Categorical(frame["Dataset"], ["MNIST", "CIFAR-10"])
    frame = DataFrame(frame)

    plot = (
        ggplot(frame)
        + aes(x="Accuracy", y="Method")
        + facet_wrap(["Dataset", "Architecture"], nrow=2, labeller="label_both")
        + geom_point(
            # aes(),
            # size=3,
            # position=position_dodge(width=0.8),
            # stroke=0.2,
        )
    )

    # plot = plot + theme(strip_text_x=element_text(size=5))

    plot.save(output_file_name + ".png", dpi=600)

plot(elements, output_file)