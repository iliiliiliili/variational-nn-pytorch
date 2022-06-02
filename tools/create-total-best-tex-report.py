import sys
import os
import json


models_root = "./models" if len(sys.argv) < 2 else sys.argv[1]
model_folders = [str(f.path) for f in os.scandir(models_root) if f.is_dir()]
output_file = (
    "tools/tex-reports/total-report.tex" if len(sys.argv) < 3 else sys.argv[2]
)
template_file = (
    "tools/tex-report-template-long.tex" if len(sys.argv) < 4 else sys.argv[3]
)

captions = "    Dataset & Model & Type & Activation & ActMode & optimizer & BN/DO & Accuracy\\\\ [0.5ex] \n        \\hline"
mode = "|c c c c c c c c|"

with open(template_file, "r") as f:
    template = f.read()


def get_model_group(model_name):

    groups = [
        'densenet',
        'resnet',
        'vgg',
        'base',
    ]

    for g in groups:
        if g in model_name:
            return g

    return model_name


def parse_model_params(path):
    with open(path + "/params.json", "r") as f:
        params = json.load(f)
        result = [
            params["dataset"].replace('_', ' '),
            params["model_name"].replace('_', ' '),
            params["type"].replace('_', ' '),
            params["activation"].replace('_', ' '),
            params["activation_mode"].replace('_', ' '),
            params["optimizer"].replace('_', ' '),
            params["regularization_type"].replace('_', ' '),
        ]

        return result


elements = []

for path in model_folders:

    if os.path.exists(path + "/best/description.json") and os.path.exists(
        path + "/params.json"
    ):
        with open(path + "/best/description.json", "r") as f:
            try:
                best = json.load(f)

            except Exception:
                continue

            elements.append(
                [
                    *parse_model_params(path),
                    str(best["result"]),
                ]
            )

print(f"Found {len(elements)} models")
elements.sort(key=lambda x: (x[0], get_model_group(x[1]), x[-1]), reverse=True)

table = [captions]

last_dataset = None
last_model = None
for element in elements:
    if element[0] != last_dataset:
        table.append("\\hline\\hline")
        last_dataset = element[0]
        last_model = get_model_group(element[1])
    elif get_model_group(element[1]) != last_model:
        table.append("\\hline")
        last_model = get_model_group(element[1])

    table.append(" & ".join(element) + " \\\\")

table[-1] += " [1ex]"

table = "\n        ".join(table)

template = template.replace("<MODE>", mode)
template = template.replace("<TABLE>", table)

with open(output_file, "w") as f:
    f.write(template)
