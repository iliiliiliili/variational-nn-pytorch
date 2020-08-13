from typing import Callable, Dict, List, TypeVar, Union
from inspect import signature


T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')


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
