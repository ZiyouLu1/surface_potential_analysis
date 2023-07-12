from __future__ import annotations

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

from .constants import FERMI_WAVEVECTOR
from .s4_wavepacket import get_hydrogen_energy_difference
from .s5_overlap import get_overlap

if TYPE_CHECKING:
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


def get_fcc_hcp_gamma() -> np.complex_:
    overlap = get_overlap(0, 1)
    interpolator_0_1 = get_overlap_momentum_interpolator_flat(overlap)

    def overlap_function(
        q: np.ndarray[_S0Inv, np.dtype[np.float_]]
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        return get_angle_averaged_diagonal_overlap_function(interpolator_0_1, q).astype(  # type: ignore[no-any-return]
            np.complex_
        )

    return calculate_hermitian_gamma_potential_integral(
        FERMI_WAVEVECTOR, overlap_function
    )


def get_rate_simple_equation(
    temperature: np.ndarray[_S0Inv, np.dtype[np.float_]]
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    omega = float(get_hydrogen_energy_difference(0, 1))
    temperature_flat = temperature.ravel()
    temperature_dep_integral = np.array(
        [
            calculate_hermitian_gamma_occupation_integral(
                omega, FERMI_WAVEVECTOR, Boltzmann * t
            )
            for t in temperature_flat
        ]
    )
    temperature_dep_integral2 = np.array(
        [
            calculate_hermitian_gamma_occupation_integral(
                -omega, FERMI_WAVEVECTOR, Boltzmann * t
            )
            for t in temperature_flat
        ]
    )
    fcc_hcp_gamma = float(np.real_if_close(get_fcc_hcp_gamma()))
    return (  # type: ignore[no-any-return]
        (temperature_dep_integral + temperature_dep_integral2) * (3 * (fcc_hcp_gamma))
    ).reshape(temperature.shape)


def calculate_tight_binding_fast_and_slow_rates() -> None:
    pass
