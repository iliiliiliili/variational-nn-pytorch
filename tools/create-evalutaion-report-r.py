import sys
import os
import json
from metrics import AverageMetric

models_root = "./models" if len(sys.argv) < 2 else sys.argv[1]
model_folders = [str(f.path) for f in os.scandir(models_root) if f.is_dir()]
output_file = (
    "tools/r-reports/evaluation-report.r" if len(sys.argv) < 3 else sys.argv[2]
)
template_file = (
    "tools/r-evaluation-report-template.r"
    if len(sys.argv) < 4
    else sys.argv[3]
)

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

        network_type = "Normal"
        regularization_type = ""
        activation_mode = "mean"

        if "densenet" in params["network_name"]:
            regularization_type += "BN E"

        if "dropout" in params["network_name"]:
            network_type = "Dropout"
            regularization_type += "DO"

            if "dropout_type" in params:
                dropout_type = params["dropout_type"]
                regularization_type += " " + dropout_type

        elif "classic" in params["network_name"]:
            network_type = "Normal"
        elif "vnn" in params["network_name"]:

            network_type = "Var"

            if "global_std_mode" in params:
                network_type = {
                    "none": "Var",
                    "replace": "VarGStd",
                    "multiply": "VarDStd",
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
            activation_mode.replace("+", " ")
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


elements = {
    "attacked": {},
    "bar_occluded": {},
    "gaussian": {},
    "normal": {"validation": []},
    "randomly_occluded": {},
    "randomly_swapped": {},
    "uniform": {},
}


def process_attacked(path, model, dataset):
    try:
        with open(
            path + "/best/results/eval_attacked_validation.txt", "r"
        ) as f:
            lines = f.readlines()

            for line in lines:
                _, attack, _, result, _, fps = line.split(" ")

                if attack not in elements["attacked"]:
                    elements["attacked"][attack] = []

                elements["attacked"][attack].append(
                    {
                        "model": model,
                        "result": result,
                        "fps": fps,
                        "dataset": dataset,
                    }
                )
    except Exception:
        print("\nNo attacked results")


def process_randomly_occluded(path, model, dataset):
    try:
        with open(
            path + "/best/results/eval_randomly_occluded_validation.txt", "r"
        ) as f:
            lines = f.readlines()

            for line in lines:
                _, chance, _, result = line.split(" ")

                if chance not in elements["randomly_occluded"]:
                    elements["randomly_occluded"][chance] = []

                elements["randomly_occluded"][chance].append(
                    {"model": model, "result": result, "dataset": dataset}
                )
    except FileNotFoundError:
        print("\nNo randomly_occluded results")


def process_randomly_swapped(path, model, dataset):
    try:
        with open(
            path + "/best/results/eval_randomly_swapped_validation.txt", "r"
        ) as f:
            lines = f.readlines()

            for line in lines:
                _, swaps, _, result = line.split(" ")

                if swaps not in elements["randomly_swapped"]:
                    elements["randomly_swapped"][swaps] = []

                elements["randomly_swapped"][swaps].append(
                    {"model": model, "result": result, "dataset": dataset}
                )
    except FileNotFoundError:
        print("\nNo randomly_occluded results")


def process_bar_occluded(path, model, dataset):
    try:
        with open(
            path + "/best/results/eval_bar_occluded_validation.txt", "r"
        ) as f:
            lines = f.readlines()

            for line in lines:
                _, bound_width, bound_height, _, result = line.split(" ")

                bounds = bound_width + " " + bound_height

                if bounds not in elements["bar_occluded"]:
                    elements["bar_occluded"][bounds] = []

                elements["bar_occluded"][bounds].append(
                    {"model": model, "result": result, "dataset": dataset}
                )
    except FileNotFoundError:
        print("\nNo bar_occluded results")


def process_gaussian(path, model, dataset):
    try:
        with open(
            path + "/best/results/eval_gaussian_validation.txt", "r"
        ) as f:
            lines = f.readlines()

            for line in lines:
                _, std, _, result = line.split(" ")

                if std not in elements["gaussian"]:
                    elements["gaussian"][std] = []

                elements["gaussian"][std].append(
                    {"model": model, "result": result, "dataset": dataset}
                )
    except FileNotFoundError:
        print("\nNo gaussian results")


def process_uniform(path, model, dataset):
    try:
        with open(
            path + "/best/results/eval_uniform_validation.txt", "r"
        ) as f:
            lines = f.readlines()

            for line in lines:
                _, std, _, result = line.split(" ")

                if std not in elements["uniform"]:
                    elements["uniform"][std] = []

                elements["uniform"][std].append(
                    {"model": model, "result": result, "dataset": dataset}
                )
    except FileNotFoundError:
        print("\nNo uniform results")


def process_normal(path, model, dataset):
    try:
        with open(path + "/best/results/eval_normal_validation.txt", "r") as f:
            lines = f.readlines()

            for line in lines:
                (result,) = line.split(" ")

                elements["normal"]["validation"].append(
                    {"model": model, "result": result, "dataset": dataset}
                )
    except Exception:
        print("\nNo normal results")


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

            (
                dataset_name,
                network_name,
                network_type,
                activation,
                optimizer,
                regularization_type,
            ) = parse_model_params(path)

            group = get_model_group(network_name)
            model = network_type + " " + group

            if "cifar10" in dataset_name:
                dataset_name = "cifar10"

            process_attacked(path, model, dataset_name)
            process_bar_occluded(path, model, dataset_name)
            process_gaussian(path, model, dataset_name)
            process_normal(path, model, dataset_name)
            process_randomly_occluded(path, model, dataset_name)
            process_randomly_swapped(path, model, dataset_name)
            process_uniform(path, model, dataset_name)

            print()

reduced_elements = {}


for eval_type, data0 in elements.items():
    reduced_elements[eval_type] = {}
    for eval_sub_type, data1 in data0.items():
        reduced_elements[eval_type][eval_sub_type] = []
        groups = {}

        for description in data1:
            group_name = description["model"] + description["dataset"]

            if group_name not in groups:
                groups[group_name] = []

            groups[group_name].append(description)

        for group, values in groups.items():

            results_metric = AverageMetric()
            fps_metric = None

            for value in values:
                results_metric.update(float(value["result"]))

                if "fps" in value:
                    if fps_metric is None:
                        fps_metric = AverageMetric()

                    fps_metric.update(float(value["fps"]))

            description = {
                "model": values[0]["model"],
                "dataset": values[0]["dataset"],
                "result": results_metric.get(),
            }

            if fps_metric is not None:
                description["fps"] = fps_metric.get()

            reduced_elements[eval_type][eval_sub_type].append(description)
        reduced_elements[eval_type][eval_sub_type].sort(
            key=lambda x: (x["model"], x["dataset"])
        )


print()

models = []
eval_types = []
eval_sub_types = []
values = []
datasets = []

i = 0
split = 100

for eval_type, data0 in reduced_elements.items():
    for eval_sub_type, data1 in data0.items():
        for description in data1:
            i += 1
            models.append(
                ("\n" if i % split == 0 else "")
                + "'"
                + description["model"]
                + "'"
            )
            eval_types.append(
                ("\n" if i % split == 0 else "") + "'" + eval_type + "'"
            )
            eval_sub_types.append(
                ("\n" if i % split == 0 else "") + "'" + eval_sub_type + "'"
            )
            values.append(
                ("\n" if i % split == 0 else "")
                + "%.4f" % description["result"]
            )
            datasets.append(
                ("\n" if i % split == 0 else "")
                + "'"
                + description["dataset"]
                + "'"
            )

template = template.replace("<MODELS>", ", ".join(models))
template = template.replace("<EVALTYPES>", ", ".join(eval_types))
template = template.replace("<EVALSUBTYPES>", ", ".join(eval_sub_types))
template = template.replace("<VALUES>", ", ".join(values))
template = template.replace("<DATASETS>", ", ".join(datasets))

with open(output_file, "w") as f:
    f.write(template)
