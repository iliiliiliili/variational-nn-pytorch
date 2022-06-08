

import fire
from main import train


def create_vnn_model_kwargs(epochs):

    result = []

    for samples in [1, 10, 100]:
        for batch in [2, 20]:
            for epochs in [epochs]:
                for optimizer in ["Adam", "SGD"]:
                    for activation in ["lrelu", "relu", "tanh"]:
                        for learning_rate in [1e-3, 5e-5]:
                            for activation_mode in ["mean", "mean+std", "mean+end", "end", "none"]:
                                for global_std_mode in ["none", "replace", "multiply"]:
                                    batch_norm_mode = activation_mode
                                    start_global_std = 1 if global_std_mode != "none" else None
                                    end_global_std = 0.5 if global_std_mode != "none" else None

                                    current_activation = activation

                                    if len(activation_mode.split("+")) > 1:
                                        current_activation = " ".join([current_activation] * len(activation_mode.split("+")))

                                    model_suffix = (
                                        "s" + str(samples) +
                                        "b" + str(batch) +
                                        "lr" + str(learning_rate).replace(".", "") + "-" +
                                        "o" + str(optimizer) + "-" +
                                        "a" + str(activation) + "-" +
                                        "am" + str(activation_mode) + "-" +
                                        "gs" + str(global_std_mode)
                                    )

                                    kwargs = {
                                        "network_type": "vnn",
                                        "model_suffix": model_suffix,
                                        "epochs": epochs,
                                        "batch": batch,
                                        "samples": samples,
                                        "optimizer": optimizer,
                                        "activation": current_activation,
                                        "activation_mode": activation_mode,
                                        "batch_norm_mode": batch_norm_mode,
                                        "optimizer_lr": learning_rate,
                                        "start_global_std": start_global_std,
                                        "end_global_std": end_global_std,
                                    }

                                    result.append(kwargs)

    return result


def create_reduced_vnn_model_kwargs(epochs):

    result = []

    for samples in [1, 10]:
        for batch in [16]:
            for epochs in [epochs]:
                for optimizer in ["Adam"]:
                    for activation in ["lrelu"]:
                        for learning_rate in [1e-4]:
                            for activation_mode, global_std_mode in [
                                ["mean", "none"], 
                                ["mean+std", "multiply"],
                                ["mean+end", "replace"],
                                ["none", "none"]
                            ]:
                                batch_norm_mode = activation_mode
                                start_global_std = 1 if global_std_mode != "none" else None
                                end_global_std = 0.5 if global_std_mode != "none" else None

                                current_activation = activation
                                activation_name = activation

                                if len(activation_mode.split("+")) > 1:
                                    current_activation = " ".join([current_activation] * len(activation_mode.split("+")))

                                if activation_mode == "none":
                                    activation_name = "none"

                                model_suffix = (
                                    "s" + str(samples) +
                                    "b" + str(batch) +
                                    "lr" + str(learning_rate).replace(".", "") + "-" +
                                    "o" + str(optimizer) + "-" +
                                    "a" + str(activation_name) + "-" +
                                    "am" + str(activation_mode) + "-" +
                                    "gs" + str(global_std_mode)
                                )

                                kwargs = {
                                    "network_type": "vnn",
                                    "model_suffix": model_suffix,
                                    "epochs": epochs,
                                    "batch": batch,
                                    "samples": samples,
                                    "optimizer": optimizer,
                                    "activation": current_activation,
                                    "activation_mode": activation_mode,
                                    "batch_norm_mode": batch_norm_mode,
                                    "optimizer_lr": learning_rate,
                                    "start_global_std": start_global_std,
                                    "end_global_std": end_global_std,
                                }

                                result.append(kwargs)

    return result


def create_classic_model_kwargs(epochs):

    result = []

    for samples in [1]:
        for batch in [16]:
            for epochs in [epochs]:
                for optimizer in ["Adam"]:
                    for activation in ["lrelu"]:
                        for learning_rate in [1e-4]:
                            for use_batch_norm in [True]:
                                    model_suffix = (
                                        "s" + str(samples) +
                                        "b" + str(batch) +
                                        "lr" + str(learning_rate).replace(".", "") + "-" +
                                        "o" + str(optimizer) + "-" +
                                        "a" + str(activation) + 
                                        ("-bn" if use_batch_norm else "")
                                    )

                                    kwargs = {
                                        "network_type": "classic",
                                        "model_suffix": model_suffix,
                                        "epochs": epochs,
                                        "batch": batch,
                                        "samples": samples,
                                        "optimizer": optimizer,
                                        "activation": activation,
                                        "use_batch_norm": use_batch_norm,
                                        "optimizer_lr": learning_rate,
                                    }

                                    result.append(kwargs)

    return result


def create_ensemble_model_kwargs(epochs):

    result = []

    for num_ensemble in [10]:
        for batch in [5]:
            for epochs in [epochs]:
                for optimizer in ["Adam"]:
                    for activation in ["lrelu"]:
                        for learning_rate in [1e-4]:
                            for use_batch_norm in [True]:
                                for prior_scale in [0, 1]:

                                    model_suffix = (
                                        "s" + str(num_ensemble) +
                                        "b" + str(batch) +
                                        "lr" + str(learning_rate).replace(".", "") + "-" +
                                        "o" + str(optimizer) + "-" +
                                        "a" + str(activation) +
                                        "ps" + str(prior_scale) +
                                        ("-bn" if use_batch_norm else "")
                                    )

                                    kwargs = {
                                        "network_type": "ensemble",
                                        "model_suffix": model_suffix,
                                        "epochs": epochs,
                                        "batch": batch,
                                        "samples": 1,
                                        "num_ensemble": num_ensemble,
                                        "optimizer": optimizer,
                                        "activation": activation,
                                        "use_batch_norm": use_batch_norm,
                                        "optimizer_lr": learning_rate,
                                        "prior_scale": prior_scale,
                                    }

                                    result.append(kwargs)

    return result


def create_dropout_model_kwargs(epochs):

    result = []

    for samples in [1, 10]:
        for dropout_probability in [0.05, 0.1, 0.2]:
            for batch in [16]:
                for epochs in [epochs]:
                    for optimizer in ["Adam"]:
                        for activation in ["lrelu"]:
                            for learning_rate in [1e-4]:
                                model_suffix = (
                                    "dp" + str(dropout_probability).replace(".", "") +
                                    "s" + str(samples) +
                                    "b" + str(batch) +
                                    "lr" + str(learning_rate).replace(".", "") + "-" +
                                    "o" + str(optimizer) + "-" +
                                    "a" + str(activation)
                                )

                                kwargs = {
                                    "network_type": "dropout",
                                    "model_suffix": model_suffix,
                                    "epochs": epochs,
                                    "batch": batch,
                                    "samples": samples,
                                    "dropout_probability": dropout_probability,
                                    "optimizer": optimizer,
                                    "activation": activation,
                                    "optimizer_lr": learning_rate,
                                }

                                result.append(kwargs)

    return result


def create_bbb_model_kwargs(epochs):

    result = []

    for samples in [1, 10]:
        for sigma_0 in [1, 2, 10, 100]:
            for batch in [16]:
                for epochs in [epochs]:
                    for optimizer in ["Adam"]:
                        for activation in ["lrelu"]:
                            for learning_rate in [1e-4]:
                                for use_batch_norm in [True]:
                                    model_suffix = (
                                        "sigma" + str(sigma_0).replace(".", "") +
                                        "s" + str(samples) +
                                        "b" + str(batch) +
                                        "lr" + str(learning_rate).replace(".", "") + "-" +
                                        "o" + str(optimizer) + "-" +
                                        "a" + str(activation)
                                    )

                                    kwargs = {
                                        "network_type": "bbb",
                                        "loss": "bbb",
                                        "model_suffix": model_suffix,
                                        "epochs": epochs,
                                        "batch": batch,
                                        "samples": samples,
                                        "index_scale": sigma_0,
                                        "loss_sigma_0": sigma_0,
                                        "optimizer": optimizer,
                                        "activation": activation,
                                        "optimizer_lr": learning_rate,
                                        "use_batch_norm": use_batch_norm,
                                    }

                                    result.append(kwargs)

    return result


def create_hypermodel_model_kwargs(epochs):

    result = []

    for learning_rate in [1e-4, 1e-3]:
        for samples in [5, 10]:
            for batch in [2, 10, 20]:
                for index_scale in [1, 2, 10, 100]:
                    for index_dim in [10, 5, 1]:
                        for epochs in [epochs]:
                            for optimizer in ["Adam"]:
                                for activation in ["lrelu"]:
                                    for use_batch_norm in [True]:
                                        model_suffix = (
                                            "s" + str(samples) +
                                            "b" + str(batch) +
                                            "is" + str(index_scale) +
                                            "id" + str(index_dim) +
                                            "lr" + str(learning_rate).replace(".", "") + "-" +
                                            "o" + str(optimizer) + "-" +
                                            "a" + str(activation)
                                        )

                                        kwargs = {
                                            "network_type": "hypermodel",
                                            "model_suffix": model_suffix,
                                            "index_dim": index_dim,
                                            "epochs": epochs,
                                            "batch": batch,
                                            "samples": samples,
                                            "index_scale": index_scale,
                                            "optimizer": optimizer,
                                            "activation": activation,
                                            "optimizer_lr": learning_rate,
                                            "use_batch_norm": use_batch_norm,
                                        }

                                        result.append(kwargs)

    return result


def create_models(datasets, network_types):

    if network_types is None or network_types == "all":
        network_types = ["vnn", "classic", "ensemble", "dropout", "bbb", "hypermodel"]

    kwarg_creators = {
        "vnn": create_reduced_vnn_model_kwargs,
        "classic": create_classic_model_kwargs,
        "ensemble": create_ensemble_model_kwargs,
        "dropout": create_dropout_model_kwargs,
        "bbb": create_bbb_model_kwargs,
        "hypermodel": create_hypermodel_model_kwargs,
    }

    networks = {
        "mnist": [
            "mnist_mini_base",
            "mnist_mini2_base",
            # "mnist_conv_max",
            "mnist_mlp",
        ],
        "cifar10": [
            "cifar10_base",
            # "cifar10_mini_base",
            "resnet_18",
            # "resnet_101",
            # "vgg_13",
            # "densenet2",
        ]
    }

    dataset_names = {
        "mnist": "mnist",
        "cifar10": "cifar10_n2",
    }
    epochs = {
        "mnist": 5,
        "cifar10": 100,
    }

    all_models = []

    for dataset in datasets:
        dataset_name = dataset_names[dataset]
        network_names = networks[dataset]

        for network_name in network_names:
            for network_type in network_types:
                kwargs_list = kwarg_creators[network_type](epochs[dataset])
                for kwargs in kwargs_list:
                    all_models.append((network_name, dataset_name, kwargs))
    
    print("Total models:", len(all_models))
    return all_models


def run(network_types="all", id=0, gpu_capacity=4, total_devices=4, datasets=["mnist"]):

    device_id = id % total_devices
    i = id

    all_models = create_models(datasets, network_types)
    
    while i < len(all_models):
        network_name, dataset_name, kwargs = all_models[i]

        try:
            train(network_name=network_name, dataset_name=dataset_name, allow_retrain=False, device="cuda:" + str(device_id), **kwargs)
        except Exception as e:
            print("ERROR:", e)
            with open("modeling_errors.txt", "a") as f:
                description = {
                    "network_name": network_name,
                    "dataset_name": dataset_name,
                    "allow_retrain": False,
                    "device": "cuda:" + str(device_id),
                    **kwargs
                }
                f.write(str(e) + " @ " + str(i) + " @ " + str(network_name) + str(description) + "\n")

        i += gpu_capacity * total_devices


    print()


def run_indexed(network_types="all", index=0, output_dir="./models", datasets=["cifar10"]):

    i = index

    all_models = create_models(datasets, network_types)
    
    network_name, dataset_name, kwargs = all_models[i]

    try:
        train(network_name=network_name, dataset_name=dataset_name, allow_retrain=False, device="cuda", all_models_path=output_dir, **kwargs)
    except Exception as e:
        print("ERROR:", e)
        with open(f"{output_dir}/modeling_errors.txt", "a") as f:
            description = {
                "network_name": network_name,
                "dataset_name": dataset_name,
                "allow_retrain": False,
                "device": "cuda",
                **kwargs
            }
            f.write(str(e) + str(network_name) + str(description))

    print()


if __name__ == "__main__":

    fire.Fire()
