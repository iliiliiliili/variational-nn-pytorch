import os
import sys
import json

models_root = "./models" if len(sys.argv) < 2 else sys.argv[1]
model_folders = [str(f.path) for f in os.scandir(models_root) if f.is_dir()]

datasets = [
    "cifar10_n2",
    "cifar10_n",
    "cifar10",
    "mnist_0_1",
    "mnist",
]

model_names = [
    "base_vnn",
    "base_classic",
    "base_dropout",
    "auto_encoder_base_vnn",
    "base_vnn",
    "base_classic",
    "base_dropout",
    "resnet_vnn_18",
    "resnet_vnn_34",
    "resnet_vnn_50",
    "resnet_vnn_101",
    "resnet_vnn_152",
    "resnet_classic_18",
    "resnet_classic_34",
    "resnet_classic_50",
    "resnet_classic_101",
    "resnet_classic_152",
    "resnet_dropout_18",
    "resnet_dropout_34",
    "resnet_dropout_50",
    "resnet_dropout_101",
    "resnet_dropout_152",
    "vgg_vnn_11",
    "vgg_vnn_13",
    "vgg_vnn_16",
    "vgg_vnn_19",
    "vgg_classic_11",
    "vgg_classic_13",
    "vgg_classic_16",
    "vgg_classic_19",
    "vgg_dropout_11",
    "vgg_dropout_13",
    "vgg_dropout_16",
    "vgg_dropout_19",
    "densenet_pure",
    "densenet1_classic",
    "densenet1_vnn",
    "densenet2_classic",
    "densenet2_vnn",
    "resnet_pure_50",
    "vgg_pure_16",
]

default_bn_models = [
    "resnet_vnn_18",
    "resnet_vnn_34",
    "resnet_vnn_50",
    "resnet_vnn_101",
    "resnet_vnn_152",
    "resnet_classic_18",
    "resnet_classic_34",
    "resnet_classic_50",
    "resnet_classic_101",
    "resnet_classic_152",
    "resnet_dropout_18",
    "resnet_dropout_34",
    "resnet_dropout_50",
    "resnet_dropout_101",
    "resnet_dropout_152",
    "densenet_pure",
    "densenet1_classic",
    "densenet1_vnn",
    "densenet2_classic",
    "densenet2_vnn",
    "resnet_pure_50",
]

activations = [
    "leacky_relu",
    "relu6",
    "relu",
    "sigmoid",
    "tanh",
]

optimizers = [
    "SGD",
    "Adam",
]

regularization_types = [
    ["bn", "BN"],
    ["dropout_alpha", "DoA"],
    ["dropout_feature_alpha", "DoFA"],
    ["dropout_standart", "Do"],
]

vnn_types = [
    ["global", "VarGSTD"],
    ["decaying", "VarDSTD"]
]

for name in model_folders:

    full_name = name

    dataset = ""
    model_name = ""
    activation = ""
    optimizer = "SGD"
    regularization_type = "-"
    nn_type = "Normal"

    for d in datasets:
        if d in name:
            dataset = d
            name = name.replace(d, "")
            break

    for n in model_names:
        if n in name:
            model_name = n
            name = name.replace(n, "")
            break

    for a in activations:
        if a in name:
            activation = a
            name = name.replace(a, "")
            break

    for op in optimizers:
        if op in name:
            optimizer = op
            name = name.replace(op, "")
            break

    for b in regularization_types:
        if b[0] in name:
            regularization_type = b[1]
            name = name.replace(b[0], "")
            break

    if (model_name in default_bn_models) and (regularization_type == "-"):
        regularization_type = "BN"

    if "vnn" in model_name:
        nn_type = "Var"

        for v in vnn_types:
            if v[0] in name:
                nn_type = v[1]

    model_name = model_name.replace('_classic', '')
    model_name = model_name.replace('_dropout', '')
    model_name = model_name.replace('_vnn', '')

    result = {
        "dataset": dataset,
        "model_name": model_name,
        "type": nn_type,
        "activation": activation,
        "optimizer": optimizer,
        "regularization_type": regularization_type,
    }

    print(result)

    with open(full_name + "/params.json", "w") as f:
        json.dump(result, f)
