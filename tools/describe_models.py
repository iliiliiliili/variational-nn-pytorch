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
    "mini_base_vnn",
    "mini_base_classic",
    "mini_base_dropout",
    "mini2_base_vnn",
    "mini2_base_classic",
    "mini2_base_dropout",
    "double_linear_vnn",
    "double_linear_classic",
    "double_linear_dropout",
    "mlp_vnn",
    "mlp_classic",
    "mlp_dropout",
    "base_0vnn",
    "base_1vnn",
    "base_smvnn",
    "base_ssvnn",
    "auto_encoder_base_vnn",
    "base_vnn",
    "base_classic",
    "base_dropout",
    "mini_base_vnn",
    "mini_base_classic",
    "mini_base_dropout",
    "resnet_vnn_18",
    "ResNet18",
    "resnet_vnn_34",
    "ResNet34",
    "resnet_vnn_50",
    "ResNet50",
    "resnet_vnn_101",
    "ResNet101",
    "resnet_vnn_152",
    "ResNet152",
    "resnet_classic_18",
    "ResNet18",
    "resnet_classic_34",
    "ResNet34",
    "resnet_classic_50",
    "ResNet50",
    "resnet_classic_101",
    "ResNet101",
    "resnet_classic_152",
    "ResNet152",
    "resnet_dropout_18",
    "ResNet18",
    "resnet_dropout_34",
    "ResNet34",
    "resnet_dropout_50",
    "ResNet50",
    "resnet_dropout_101",
    "ResNet101",
    "resnet_dropout_152",
    "ResNet152",
    "vgg_vnn_11",
    "VGG11",
    "vgg_vnn_13",
    "VGG13",
    "vgg_vnn_16",
    "VGG16",
    "vgg_vnn_19",
    "VGG19",
    "VGG11",
    "VGG13",
    "VGG16",
    "VGG19",
    "VGG11",
    "VGG13",
    "VGG16",
    "VGG19",
    "densenet_pure",
    "densenet1_classic",
    "densenet1_vnn",
    "densenet2_classic",
    "densenet2_vnn",
    "densenet2_dropout",
    "resnet_pure_50",
    "vgg_pure_16",
    "vgg_pure_19",
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
    ["bn_bmdm", "BNMean"],
    ["bn_bmds", "BNStd"],
    ["bn_bmde", "BNEnd"],
    ["bn_bmdms", "BNMeanStd"],
    ["bn_bmdse", "BNStdEnd"],
    ["bn_bmdmse", "BNMeanStdEnd"],
    ["bn", "BN"],
    ["dropout_alpha", "DoA"],
    ["dropout_feature_alpha", "DoFA"],
    ["dropout_standart", "Do"],
]

vnn_types = [
    ["global", "VarGSTD"],
    ["decaying", "VarDSTD"],
    ["varp", "Var"],
    ["varg", "VarGSTD"],
    ["vard", "VarDSTD"],
]

complex_activations = [
    ["noeact_amdm", "-", "-"],
    ["relu_amdm", "Relu", "Mean"],
    ["relu_amds", "Relu", "Std"],
    ["relu_amde", "Relu", "End"],
    ["relu_amdms", "Relu Relu", "Mean+Std"],
    ["ars_amdms", "Relu Sigmoid", "Mean+Std"],
    ["art_amdm", "Relu Tanh", "Mean+Std"],
    ["arrr_amdmse", "Relu Relu Relu", "Mean+Std+End"],
    ["arsr_amdmse", "Relu Sigmoid Relu", "Mean+Std+End"],
    ["artr_amdmse", "Relu Tanh Relu", "Mean+Std+End"],
    ["relu6_amdm", "Relu6", "Mean"],
    ["relu6_amds", "Relu6", "Std"],
    ["relu6_amde", "Relu6", "End"],
    ["relu6_amdms", "Relu6 Relu6", "Mean+Std"],
    ["ar6s_amdms", "Relu6 Sigmoid", "Mean+Std"],
    ["ar6t_amdm", "Relu6 Tanh", "Mean+Std"],
    ["ar6r6r6_amdmse", "Relu6 Relu6 Relu6", "Mean+Std+End"],
    ["ar6sr6_amdmse", "Relu6 Sigmoid Relu6", "Mean+Std+End"],
    ["ar6tr6_amdmse", "Relu6 Tanh Relu6", "Mean+Std+End"],
    ["sigmoid_amdm", "Sigmoid", "Mean"],
    ["sigmoid_amds", "Sigmoid", "Std"],
    ["sigmoid_amde", "Sigmoid", "End"],
    ["ass_amdms", "Sigmoid Sigmoid", "Mean+Std"],
    ["ast_amdms", "Sigmoid Tanh", "Mean+Std"],
    ["asss_amdmse", "Sigmoid Sigmoid Sigmoid", "Mean+Std+End"],
    ["asts_amdmse", "Sigmoid Tanh Sigmoid", "Mean+Std+End"],
    ["tanh_amdm", "Tanh", "Mean"],
    ["tanh_amds", "Tanh", "Std"],
    ["tanh_amde", "Tanh", "End"],
    ["att_amdms", "Tanh Tanh", "Mean+Std"],
    ["ats_amdms", "Tanh Sigmoid", "Mean+Std"],
    ["attt_amdmse", "Tanh Tanh Tanh", "Mean+Std+End"],
    ["atst_amdmse", "Tanh Sigmoid Tanh", "Mean+Std+End"],
    ["leacky_relu_amdm", "LRelu", "Mean"],
    ["leacky_relu_amds", "LRelu", "Std"],
    ["leacky_relu_amde", "LRelu", "End"],
    ["alrlr_amdms", "LRelu LRelu", "Mean+Std"],
    ["alrs_amdms", "LRelu Sigmoid", "Mean+Std"],
    ["alr_amdm", "LRelu Tanh", "Mean+Std"],
    ["alrlrlr_amdmse", "LRelu LRelu LRelu", "Mean+Std+End",],
    ["alrslr_amdmse", "LRelu Sigmoid LRelu", "Mean+Std+End"],
    ["alrtlr_amdmse", "LRelu Tanh LRelu", "Mean+Std+End"],
    ["arr_amdms", "Relu Relu", "Mean+Std"],
    ["ar6r6_amdms", "Relu6 Relu6", "Mean+Std"],
    ["ass_amdms", "Sigmoid Sigmoid", "Mean+Std"],
    ["att_amdms", "Tanh Tanh", "Mean+Std"],
    ["alrlr_amdms", "lrelu lrelu", "Mean+Std"],
    ["noact", "-", "-"],
]

for name in model_folders:

    full_name = name

    dataset = ""
    model_name = ""
    activation = ""
    optimizer = "SGD"
    regularization_type = "-"
    nn_type = "Normal"
    activation_mode = "Mean"

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

    model_name = model_name.replace("_classic", "")
    model_name = model_name.replace("_dropout", "")
    model_name = model_name.replace("_vnn", "")

    for ca in complex_activations:
        if ca[0] in model_name:
            activation = ca[1]
            activation_mode = ca[2]
            break

    result = {
        "dataset": dataset,
        "model_name": model_name,
        "type": nn_type,
        "activation": activation,
        "activation_mode": activation_mode,
        "optimizer": optimizer,
        "regularization_type": regularization_type,
    }

    print(result)

    with open(full_name + "/params.json", "w") as f:
        json.dump(result, f)
