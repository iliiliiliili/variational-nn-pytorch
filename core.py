from typing import Callable, Dict, List, TypeVar, Union
from inspect import signature
import torch
from torch import flatten, nn


T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')


class Flatten(nn.Module):

    def __init__(self, start_dim) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=start_dim)

    def forward(self, input):

        if isinstance(input, tuple):
            result = tuple(self.flatten(x) for x in input)
        else:
            result = self.flatten(input)

        return result


def rename_dict(dict: dict, func: Callable[[str], str]):
    result = {}

    for key, val in dict.items():
        result[func(key)] = val

    return result


def take(dict: dict, names: List[str]):
    result = {}

    for name in names:
        if name in dict:
            result[name] = dict[name]

    return result


def give(dict: dict, names: List[str]):
    given = {}
    left = {}

    for key, val in dict.items():
        if key in names:
            given[key] = val
        else:
            left[key] = val

    return given, left


def filter_out_dict(dict: dict, names: List[str]):
    left = {}

    for key, val in dict.items():
        if key not in names:
            left[key] = val

    return left


def map_dict(
    dict: Dict[T1, T2],
    func: Union[Callable[[T2], T3], Callable[[T2, T1], T3]]
):
    result = {}

    func_params_count = len(signature(func).parameters)

    if func_params_count == 1:
        for key, val in dict.items():
            result[key] = func(val)  # type: ignore
    elif func_params_count == 2:
        for key, val in dict.items():
            result[key] = func(val, key)  # type: ignore
    else:
        raise ValueError('Wrong reduce function')

    return result


def reduce_dict(
    dict: Dict[T1, T2],
    func: Union[Callable[[T3, T2], T3], Callable[[T3, T2, T1], T3]],
    initial_value: T3
):
    result = initial_value

    func_params_count = len(signature(func).parameters)

    if func_params_count == 2:
        for val in dict.values():
            result = func(result, val)  # type: ignore
    elif func_params_count == 3:
        for key, val in dict.items():
            result = func(result, val, key)  # type: ignore
    else:
        raise ValueError('Wrong reduce function')

    return result


def zip_dicts(dict1: Dict[T1, T2], dict2: Dict[T1, T2]):
    result = {}

    for key, val in dict1.items():

        result[key] = (val, dict2[key])

    return result


def split_by_dict(values: List[T2], dict: Dict[T1, int]):
    result = {}

    i = 0

    for key, count in dict.items():
        result[key] = values[i:i + count]
        i += count

    return result


def split_by_arrays(values: List[T2], counts: List[List[int]]):
    result = []

    i = 0

    for local_counts in counts:
        local_result = []
        for count in local_counts:
            local_result.append(values[i:i + count])
            i += count
        result.append(local_result)

    return result