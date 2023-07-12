from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from scipy.constants import Boltzmann
from surface_potential_analysis.dynamics.hermitian_gamma_integral import (
    calculate_hermitian_gamma_occupation_integral,
    calculate_hermitian_gamma_potential_integral,
)
from surface_potential_analysis.overlap.interpolation import (
    get_angle_averaged_diagonal_overlap_function,
    get_overlap_momentum_interpolator_flat,
)

from hydrogen_nickel_111.constants import FERMI_WAVEVECTOR
from hydrogen_nickel_111.s4_wavepacket import (
    get_deuterium_energy_difference,
    get_hydrogen_energy_difference,
)
from hydrogen_nickel_111.s5_overlap import get_overlap_deuterium, get_overlap_hydrogen

if TYPE_CHECKING:
    from collections.abc import Callable

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


@cache
def _get_diagonal_overlap_function_hydrogen_cached(
    i: int, j: int, offset_j: tuple[int, int]
) -> Callable[
    [np.ndarray[_S0Inv, np.dtype[np.float_]]],
    np.ndarray[_S0Inv, np.dtype[np.float_]],
]:
    overlap = get_overlap_hydrogen(i, j, (0, 0), offset_j)
    interpolator = get_overlap_momentum_interpolator_flat(overlap)

    def overlap_function(
        q: np.ndarray[_S0Inv, np.dtype[np.float_]]
    ) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
        return get_angle_averaged_diagonal_overlap_function(interpolator, q)

    return overlap_function


@cache
def _calculate_gamma_potential_integral_hydrogen_diagonal(
    i: int, j: int, offset_j: tuple[int, int]
) -> np.complex_:
    overlap_function = _get_diagonal_overlap_function_hydrogen_cached(i, j, offset_j)
    return calculate_hermitian_gamma_potential_integral(
        FERMI_WAVEVECTOR, overlap_function
    )


def calculate_gamma_potential_integral_hydrogen_diagonal(
    i: int, j: int, offset_i: tuple[int, int], offset_j: tuple[int, int]
) -> float:
    if j > i:
        (j, i) = (i, j)
        (offset_j, offset_i) = (offset_i, offset_j)

    offset_j = (offset_j[0] - offset_i[0], offset_j[1] - offset_i[1])
    if -2 < offset_j[0] < 2 and -2 < offset_j[1] < 2:  # noqa: PLR2004
        return float(
            np.real_if_close(
                _calculate_gamma_potential_integral_hydrogen_diagonal(i, j, offset_j)
            )
        )
    return 0


@cache
def calculate_gamma_occupation_integral_hydrogen_diagonal(
    i: int, j: int, temperature: float
) -> float:
    omega = get_hydrogen_energy_difference(j, i)
    return calculate_hermitian_gamma_occupation_integral(
        float(omega), FERMI_WAVEVECTOR, Boltzmann * temperature
    )


def a_function_hydrogen(
    i: int,
    j: int,
    offset_i: tuple[int, int],
    offset_j: tuple[int, int],
    temperature: float,
) -> float:
    return calculate_gamma_occupation_integral_hydrogen_diagonal(
        i, j, temperature
    ) * calculate_gamma_potential_integral_hydrogen_diagonal(i, j, offset_i, offset_j)


@cache
def _get_diagonal_overlap_function_deuterium_cached(
    i: int, j: int, offset_j: tuple[int, int]
) -> Callable[
    [np.ndarray[_S0Inv, np.dtype[np.float_]]],
    np.ndarray[_S0Inv, np.dtype[np.float_]],
]:
    overlap = get_overlap_deuterium(i, j, (0, 0), offset_j)
    interpolator = get_overlap_momentum_interpolator_flat(overlap)

    def overlap_function(
        q: np.ndarray[_S0Inv, np.dtype[np.float_]]
    ) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
        return get_angle_averaged_diagonal_overlap_function(interpolator, q)

    return overlap_function


@cache
def _calculate_gamma_potential_integral_deuterium_diagonal(
    i: int, j: int, offset_j: tuple[int, int]
) -> np.complex_:
    overlap_function = _get_diagonal_overlap_function_deuterium_cached(i, j, offset_j)
    return calculate_hermitian_gamma_potential_integral(
        FERMI_WAVEVECTOR, overlap_function
    )


def calculate_gamma_potential_integral_deuterium_diagonal(
    i: int, j: int, offset_i: tuple[int, int], offset_j: tuple[int, int]
) -> float:
    if j > i:
        (j, i) = (i, j)
        (offset_j, offset_i) = (offset_i, offset_j)

    offset_j = (offset_j[0] - offset_i[0], offset_j[1] - offset_i[1])
    if -2 < offset_j[0] < 2 and -2 < offset_j[1] < 2:  # noqa: PLR2004
        return float(
            np.real_if_close(
                _calculate_gamma_potential_integral_deuterium_diagonal(i, j, offset_j)
            )
        )
    return 0


@cache
def calculate_gamma_occupation_integral_deuterium_diagonal(
    i: int, j: int, temperature: float
) -> float:
    omega = get_deuterium_energy_difference(j, i)
    return calculate_hermitian_gamma_occupation_integral(
        float(omega), FERMI_WAVEVECTOR, Boltzmann * temperature
    )


def a_function_deuterium(
    i: int,
    j: int,
    offset_i: tuple[int, int],
    offset_j: tuple[int, int],
    temperature: float,
) -> float:
    return calculate_gamma_occupation_integral_deuterium_diagonal(
        i, j, temperature
    ) * calculate_gamma_potential_integral_deuterium_diagonal(i, j, offset_i, offset_j)
