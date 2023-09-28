from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from scipy.constants import Boltzmann
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.dynamics.hermitian_gamma_integral import (
    calculate_hermitian_gamma_occupation_integral,
    calculate_hermitian_gamma_potential_integral,
)
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
    TunnellingAMatrix,
    TunnellingJumpMatrix,
    get_a_matrix_from_jump_matrix,
    get_jump_matrix_from_function,
)
from surface_potential_analysis.dynamics.tunnelling_basis import (
    TunnellingSimulationBandsBasis,
)
from surface_potential_analysis.dynamics.util import get_hop_shift
from surface_potential_analysis.overlap.interpolation import (
    get_angle_averaged_diagonal_overlap_function,
    get_overlap_momentum_interpolator_flat,
)
from surface_potential_analysis.util.constants import FERMI_WAVEVECTOR
from surface_potential_analysis.util.decorators import npy_cached, timed
from surface_potential_analysis.wavepacket.wavepacket import (
    get_wavepackets_with_eigenvalues,
)

from .s4_wavepacket import (
    get_all_wavepackets_hydrogen,
    get_hydrogen_energy_difference,
    get_wavepacket_hydrogen,
)
from .s5_overlap import get_overlap_hydrogen
from .surface_data import get_data_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from surface_potential_analysis.basis.basis import FundamentalBasis
    from surface_potential_analysis.basis.stacked_basis import StackedBasisLike

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


@timed
def _get_diagonal_overlap_function_hydrogen(
    i: int, j: int, offset_j: tuple[int, int]
) -> Callable[
    [np.ndarray[_S0Inv, np.dtype[np.float_]]],
    np.ndarray[_S0Inv, np.dtype[np.float_]],
]:
    overlap = get_overlap_hydrogen(i, j, (0, 0), offset_j)
    n_points = np.prod(BasisUtil(get_wavepacket_hydrogen(0)["basis"]).shape[0:2])
    interpolator = get_overlap_momentum_interpolator_flat(overlap, n_points)

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


@timed
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
@timed
def calculate_gamma_occupation_integral_hydrogen_diagonal(
    i: int, j: int, temperature: float
) -> float:
    omega_j_i = get_hydrogen_energy_difference(i, j)
    return calculate_hermitian_gamma_occupation_integral(
        omega_j_i, FERMI_WAVEVECTOR["NICKEL"], Boltzmann * temperature
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


def _get_get_tunnelling_jump_matrix_hydrogen_cache(
    n_bands: _L2Inv, # type: ignore _L2Inv needs to match get_jump_matrix_hydrogen
    temperature: float,
) -> Path:
    return get_data_path(f"dynamics/jump_matrix_hydrogen_{n_bands}_{temperature}k.npy")


@npy_cached(_get_get_tunnelling_jump_matrix_hydrogen_cache, load_pickle=True)  # type: ignore[misc]
def get_jump_matrix_hydrogen(
    n_bands: _L2Inv, temperature: float
) -> TunnellingJumpMatrix[_L2Inv]:
    def jump_function(i: int, j: int, hop: int) -> float:
        offset_j = get_hop_shift(hop, 2)
        return a_function_hydrogen(
            i, j, (0, 0), (offset_j[0], offset_j[1]), temperature
        )

    basis = TunnellingSimulationBandsBasis[_L2Inv].from_wavepackets(
        get_wavepackets_with_eigenvalues(get_all_wavepackets_hydrogen(), slice(n_bands))
    )

    return get_jump_matrix_from_function(basis, jump_function)


def get_tunnelling_a_matrix_hydrogen(
    shape: tuple[_L0Inv, _L1Inv],
    n_bands: _L2Inv,
    temperature: float,
) -> TunnellingAMatrix[
    StackedBasisLike[
        FundamentalBasis[_L0Inv],
        FundamentalBasis[_L1Inv],
        TunnellingSimulationBandsBasis[_L2Inv],
    ]
]:
    jump_matrix = get_jump_matrix_hydrogen(n_bands, temperature)
    return get_a_matrix_from_jump_matrix(jump_matrix, shape, n_bands=n_bands)


def get_fey_jump_matrix_hydrogen() -> TunnellingJumpMatrix[Literal[2]]:
    """
    Get an A matrix compatible with fey's formula.

    Parameters
    ----------
    shape : tuple[_L0Inv, _L1Inv]
    temperature : float

    Returns
    -------
    TunnellingAMatrix[tuple[ FundamentalBasis[_L0Inv], FundamentalBasis[_L1Inv], TunnellingSimulationBandsBasis[Literal[2]]]]
    """

    def jump_function(i: int, j: int, hop: int) -> float:
        if i == 0 and j == 1 and (hop in [0, 2, 6]):
            return 3.02959631e08

        if i == 1 and j == 0 and (hop in [0, 1, 3]):
            return 6.55978349e08
        return 0

    axis = TunnellingSimulationBandsBasis[Literal[2]].from_wavepackets(
        get_wavepackets_with_eigenvalues(get_all_wavepackets_hydrogen(), slice(2))
    )

    return get_jump_matrix_from_function(axis, jump_function)
