import sys
import os
import json


models_root = "./models" if len(sys.argv) < 2 else sys.argv[1]
model_folders = [str(f.path) for f in os.scandir(models_root) if f.is_dir()]
output_file = (
    "tools/tex-reports/report.tex" if len(sys.argv) < 3 else sys.argv[2]
)
template_file = (
    "tools/tex-report-template-long.tex" if len(sys.argv) < 4 else sys.argv[3]
)

captions = "    Dataset & Model & Accuracy \\\\ [0.5ex] \n        \\hline"
mode = "|c c c|"

with open(template_file, "r") as f:
    template = f.read()


def parse_name(name: str):

    datasets = [
        "cifar10_n2",
        "cifar10_n",
        "cifar10",
        "mnist_0_1",
        "mnist",
    ]

    dataset = 'unknown'

    for d in datasets:
        if d in name:
            dataset = d.replace('_', ' ')
            name = name.replace(d, '')
            break

    name = name.replace('_', ' ')

    if len(name) > 60:
        name = name[:60] + "..."

    return [dataset, name]


elements = []

for path in model_folders:

    if os.path.exists(path + "/best/description.json"):
        with open(path + "/best/description.json", "r") as f:
            best = json.load(f)
            elements.append(
                [
                    *parse_name(path.split("\\")[-1].split("/")[-1]),
                    str(best["result"]),
                ]
            )
print(f"Found {len(elements)} models")
elements.sort(key=lambda x: (x[0], x[-1]), reverse=True)

table = [captions]

last_dataset = None
for element in elements:
    if element[0] != last_dataset:
        table.append("\\hline")
        last_dataset = element[0]

    table.append(" & ".join(element) + " \\\\")

table[-1] += " [1ex]"

table = "\n        ".join(table)

template = template.replace("<MODE>", mode)
template = template.replace("<TABLE>", table)

with open(output_file, "w") as f:
    f.write(template)
