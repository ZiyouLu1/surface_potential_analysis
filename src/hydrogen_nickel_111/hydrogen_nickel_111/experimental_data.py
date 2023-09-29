from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TypedDict, TypeVar

import numpy as np
import scipy.optimize

from .surface_data import get_data_path

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
_S1Inv = TypeVar("_S1Inv", bound=tuple[int, ...])
_S2Inv = TypeVar("_S2Inv", bound=tuple[int, ...])


class ExperimentData(TypedDict):
    temperature: np.ndarray[tuple[int], np.dtype[np.float_]]
    rate: np.ndarray[tuple[int], np.dtype[np.float_]]
    upper_error: np.ndarray[tuple[int], np.dtype[np.float_]]
    lower_error: np.ndarray[tuple[int], np.dtype[np.float_]]


def get_experiment_data() -> ExperimentData:
    path = get_data_path("fast_rate_experimental.json")
    with path.open("r") as f:
        out = json.load(f)
        return {
            "temperature": np.array(out["temperature"], dtype=np.float_),
            "lower_error": np.array(out["lower_error"], dtype=np.float_) * 10**10,
            "rate": np.array(out["rate"], dtype=np.float_) * 10**10,
            "upper_error": np.array(out["upper_error"], dtype=np.float_) * 10**10,
        }


RateFn = Callable[
    [np.ndarray[_S0Inv, np.dtype[np.float_]]],
    np.ndarray[_S0Inv, np.dtype[np.float_]],
]


def get_experimental_baseline_rates(
    get_rate: RateFn[Any],
) -> Callable[
    [np.ndarray[_S1Inv, np.dtype[np.float_]]],
    np.ndarray[_S1Inv, np.dtype[np.float_]],
]:
    data = get_experiment_data()
    temperatures = data["temperature"]
    rates = data["rate"] - get_rate(temperatures)

    def f(
        t: np.ndarray[_S2Inv, np.dtype[np.float_]], a: float, b: float
    ) -> np.ndarray[_S2Inv, np.dtype[np.float_]]:
        return b * np.exp(-a / t)  # type: ignore[no-any-return]

    a0 = np.log(rates[0] / rates[9]) / ((1 / temperatures[9]) - (1 / temperatures[0]))
    r0 = rates[0] * np.exp(a0 / temperatures[0])

    p_opt, _ = scipy.optimize.curve_fit(f, temperatures, rates, p0=[a0, r0])
    return lambda t: f(t, *p_opt)
