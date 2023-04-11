from typing import Literal, TypeVar

import numpy as np

from surface_potential_analysis.basis import (
    ExplicitBasis,
    MomentumBasis,
    PositionBasis,
    TruncatedBasis,
)
from surface_potential_analysis.hamiltonian import HamiltonianWithBasis
from surface_potential_analysis.hamiltonian_builder import sho_subtracted_basis
from surface_potential_analysis.sho_basis import SHOBasisConfig
from surface_potential_analysis.util import timed

from .s1_potential import get_interpolated_nickel_potential

_L0 = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
_L2 = TypeVar("_L2", bound=int)
_L3 = TypeVar("_L3", bound=int)
_L4 = TypeVar("_L4", bound=int)
_L5 = TypeVar("_L5", bound=int)


@timed
def generate_hamiltonian_sho(
    shape: tuple[_L0, _L1, _L2],
    bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    resolution: tuple[_L3, _L4, _L5],
) -> HamiltonianWithBasis[
    TruncatedBasis[_L3, MomentumBasis[_L0]],
    TruncatedBasis[_L4, MomentumBasis[_L1]],
    ExplicitBasis[_L5, PositionBasis[_L2]],
]:
    potential = get_interpolated_nickel_potential(shape)
    config: SHOBasisConfig = {
        "sho_omega": 195636899474736.66,
        "mass": 1.6735575e-27,
        "x_origin": np.array([0, 0, -1.0000000000000004e-10]),
    }
    return sho_subtracted_basis.total_surface_hamiltonian(
        potential, config, bloch_phase, resolution
    )


# def generate_sho_config():
#     data = load_interpolated_grid()
#     interpolation = as_interpolation(data)
#     mass = 1.6735575e-27
#     omega, z_offset = generate_sho_config_minimum(
#         interpolation, mass, initial_guess=1.5e14, fit_max_energy_fraction=0.3
#     )
#     print(omega, z_offset)
