from typing import List, Tuple
import torch
import numpy as np
import random
from params import (
    SEED,
    perturbed_dataset_params,
)


def evaluate_gaussian(
    evaluate,
    dataset_name,
    mean=0,
    std_start=0.1,
    std_end=1,
    count=10,
    **kwargs
):

    results = []
    step = std_end / count

    for std in np.arange(std_start, std_end + step, step):
        torch.manual_seed(SEED)

        def gaussian_noise(x):
            result = x + torch.normal(torch.ones(x.shape) * mean, std,)
            return result

        result, fps = evaluate(
            perturbed_dataset_params[dataset_name](gaussian_noise)
        )
        results.append(
            "std: " + str(round(std, 2)) + " result: " + str(result)
        )

    return "\n".join(results)


def evaluate_uniform(
    evaluate,
    dataset_name,
    mean=0,
    std_start=0.1,
    std_end=2,
    count=10,
    **kwargs
):

    results = []
    step = std_end / count

    for std in np.arange(std_start, std_end + step, step):
        torch.manual_seed(SEED)

        def gaussian_noise(x):
            result = x + mean + std * torch.rand_like(torch.ones(x.shape))
            return result

        result, fps = evaluate(
            perturbed_dataset_params[dataset_name](gaussian_noise)
        )
        results.append(
            "std: " + str(round(std, 2)) + " result: " + str(result)
        )

    return "\n".join(results)


def evaluate_bar_occluded(
    evaluate,
    dataset_name,
    size_bounds: List[Tuple[int, int]] = [(5, 10), (10, 15), (15, 20)],
    occlusion_value: float = 0.5,
    **kwargs
):

    torch.manual_seed(SEED)
    random.seed(SEED)
    results = []

    # imgs = []

    for bounds in size_bounds:

        def occluded(image):

            width, height = image.shape[1:3]

            w = random.randint(bounds[0], bounds[1])
            h = random.randint(bounds[0], bounds[1])
            x = random.randint(0, width - w)
            y = random.randint(0, height - h)

            result = image

            result[:, x : x + w, y : y + h] = occlusion_value

            # print('x', x, 'y', y, 'w', w, 'h', h)

            # if len(imgs) < 10:
            #     imgs.append(result.cpu().numpy().squeeze())
            # else:
            #     print('s')

            return result

        result, fps = evaluate(perturbed_dataset_params[dataset_name](occluded))

        results.append("bounds: " + str(bounds) + " result: " + str(result))

    return "\n".join(results)


def evaluate_randomly_occluded(
    evaluate,
    dataset_name,
    occlusion_chances: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
    occlusion_value: float = 0.5,
    **kwargs
):

    torch.manual_seed(SEED)
    random.seed(SEED)
    results = []

    # imgs = []

    for occlusion_chance in occlusion_chances:

        def occluded(image):

            saved_region = (torch.rand_like(image) > occlusion_chance) * 1.0
            occluded_region = (1 - saved_region) * occlusion_value

            result = saved_region * image + occluded_region

            # if len(imgs) < 10:
            #     imgs.append(result.cpu().numpy().squeeze())
            # else:
            #     print("s")

            return result

        result, fps = evaluate(perturbed_dataset_params[dataset_name](occluded))

        results.append(
            "occlusion_chanse: "
            + str(occlusion_chance)
            + " result: "
            + str(result)
        )

    return "\n".join(results)


def evaluate_randomly_swapped(
    evaluate,
    dataset_name,
    swap_counts: List[int] = [200, 400, 500, 600, 800],
    **kwargs
):

    torch.manual_seed(SEED)
    random.seed(SEED)
    results = []

    # imgs = []

    for swaps in swap_counts:

        def swapped(image):

            result = image
            width, height = image.shape[1:3]

            for i in range(swaps):

                x1 = random.randint(0, width - 1)
                y1 = random.randint(0, height - 1)

                x2 = random.randint(0, width - 1)
                y2 = random.randint(0, height - 1)

                temp = result[:, x1, y1]
                result[:, x1, y1] = result[:, x2, y2]
                result[:, x2, y2] = temp

            # if len(imgs) < 10:
            #     imgs.append(result.cpu().numpy().squeeze())
            # else:
            #     print("s")

            return result

        result, fps = evaluate(perturbed_dataset_params[dataset_name](swapped))

        results.append(
            "swaps: "
            + str(swaps)
            + " result: "
            + str(result)
        )

    return "\n".join(results)
