from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from scipy.constants import Boltzmann
from surface_potential_analysis.dynamics.hermitian_gamma_integral import (
    calculate_hermitian_gamma_occupation_integral,
    calculate_hermitian_gamma_potential_integral,
)
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_basis import (
    TunnellingSimulationBandsAxis,
)
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
    TunnellingAMatrix,
    get_tunnelling_a_matrix_from_function,
)
from surface_potential_analysis.overlap.interpolation import (
    get_angle_averaged_diagonal_overlap_function,
    get_overlap_momentum_interpolator_flat,
)
from surface_potential_analysis.util.constants import FERMI_WAVEVECTOR
from surface_potential_analysis.util.decorators import npy_cached

from .s4_wavepacket import (
    get_all_wavepackets_deuterium,
    get_all_wavepackets_hydrogen,
    get_deuterium_energy_difference,
    get_hydrogen_energy_difference,
)
from .s5_overlap import get_overlap_deuterium, get_overlap_hydrogen
from .surface_data import get_data_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from surface_potential_analysis.axis.axis import FundamentalAxis

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def _get_diagonal_overlap_function_hydrogen(
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
    overlap_function = _get_diagonal_overlap_function_hydrogen(i, j, offset_j)
    return calculate_hermitian_gamma_potential_integral(
        FERMI_WAVEVECTOR["NICKEL"], overlap_function
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
        float(omega), FERMI_WAVEVECTOR["NICKEL"], Boltzmann * temperature
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


def _get_diagonal_overlap_function_deuterium(
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
    overlap_function = _get_diagonal_overlap_function_deuterium(i, j, offset_j)
    return calculate_hermitian_gamma_potential_integral(
        FERMI_WAVEVECTOR["NICKEL"], overlap_function
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
        float(omega), FERMI_WAVEVECTOR["NICKEL"], Boltzmann * temperature
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


def _get_get_tunnelling_a_matrix_hydrogen_cache(
    shape: tuple[_L0Inv, _L1Inv], n_bands: _L2Inv, temperature: float
) -> Path:
    return get_data_path(
        f"dynamics/a_matrix_hydrogen_{shape[0]}_{shape[1]}_{n_bands}_{temperature}k.npy"
    )


@npy_cached(_get_get_tunnelling_a_matrix_hydrogen_cache, load_pickle=True)  # type: ignore[misc]
def get_tunnelling_a_matrix_hydrogen(
    shape: tuple[_L0Inv, _L1Inv],
    n_bands: _L2Inv,
    temperature: float,
) -> TunnellingAMatrix[
    tuple[
        FundamentalAxis[_L0Inv],
        FundamentalAxis[_L1Inv],
        TunnellingSimulationBandsAxis[_L2Inv],
    ]
]:
    def a_function(
        i: int, j: int, offset_i: tuple[int, int], offset_j: tuple[int, int]
    ) -> float:
        return a_function_hydrogen(i, j, offset_i, offset_j, temperature)

    bands_axis = TunnellingSimulationBandsAxis[_L2Inv].from_wavepackets(
        get_all_wavepackets_hydrogen()[0:n_bands]
    )

    return get_tunnelling_a_matrix_from_function(shape, bands_axis, a_function)


def _get_get_tunnelling_a_matrix_deuterium_cache(
    shape: tuple[_L0Inv, _L1Inv], n_bands: _L2Inv, temperature: float
) -> Path:
    return get_data_path(
        f"dynamics/a_matrix_deuterium_{shape[0]}_{shape[1]}_{n_bands}_{temperature}k.npy"
    )


@npy_cached(_get_get_tunnelling_a_matrix_deuterium_cache, load_pickle=True)  # type: ignore[misc]
def get_tunnelling_a_matrix_deuterium(
    shape: tuple[_L0Inv, _L1Inv],
    n_bands: _L2Inv,
    temperature: float,
) -> TunnellingAMatrix[
    tuple[
        FundamentalAxis[_L0Inv],
        FundamentalAxis[_L1Inv],
        TunnellingSimulationBandsAxis[_L2Inv],
    ]
]:
    def a_function(
        i: int, j: int, offset_i: tuple[int, int], offset_j: tuple[int, int]
    ) -> float:
        return a_function_deuterium(i, j, offset_i, offset_j, temperature)

    bands_axis = TunnellingSimulationBandsAxis[_L2Inv].from_wavepackets(
        get_all_wavepackets_deuterium()[0:n_bands]
    )
    return get_tunnelling_a_matrix_from_function(shape, bands_axis, a_function)
