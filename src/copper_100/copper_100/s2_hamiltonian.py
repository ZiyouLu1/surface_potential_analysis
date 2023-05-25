from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from surface_potential_analysis.hamiltonian_builder import sho_subtracted_basis
from surface_potential_analysis.util.decorators import timed

from copper_100.s1_potential import (
    get_interpolated_potential,
    get_interpolated_potential_relaxed,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        ExplicitBasis,
        MomentumBasis,
    )
    from surface_potential_analysis.basis_config.sho_basis import SHOBasisConfig
    from surface_potential_analysis.hamiltonian import HamiltonianWithBasis

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
    MomentumBasis[_L0, _L3], MomentumBasis[_L1, _L4], ExplicitBasis[_L2, _L5]
]:
    """
    Generate a Hamiltonian using an infinate SHO basis.

    Parameters
    ----------
    shape : tuple[_L0, _L1, _L2]
        Shape of the initial potential
    bloch_phase : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
        Bloch phase
    resolution : tuple[_L3, _L4, _L5]
        Resolution of the truncated basis in x,y

    Returns
    -------
    HamiltonianWithBasis[TruncatedBasis[_L3, MomentumBasis[_L0]], TruncatedBasis[_L4, MomentumBasis[_L1]], ExplicitBasis[_L5, PositionBasis[_L2]]]
        Hamiltonian in the specified basis
    """
    potential = get_interpolated_potential(shape)
    potential["points"] = 0.5 * (
        potential["points"] + potential["points"].swapaxes(0, 1)
    )
    config: SHOBasisConfig = {
        "sho_omega": 117905964225836.06,
        "mass": 1.6735575e-27,
        "x_origin": np.array([0, 0, -1.840551985155284e-10]),
    }
    return sho_subtracted_basis.total_surface_hamiltonian(
        potential, config, bloch_phase, resolution
    )


@timed
def generate_hamiltonian_sho_relaxed(
    shape: tuple[_L0, _L1, _L2],
    bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    resolution: tuple[_L3, _L4, _L5],
) -> HamiltonianWithBasis[
    MomentumBasis[_L0, _L3], MomentumBasis[_L1, _L4], ExplicitBasis[_L2, _L5]
]:
    """
    Generate a Hamiltonian using an infinate SHO basis.

    Parameters
    ----------
    shape : tuple[_L0, _L1, _L2]
        Shape of the initial potential
    bloch_phase : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
        Bloch phase
    resolution : tuple[_L3, _L4, _L5]
        Resolution of the truncated basis in x,y

    Returns
    -------
    HamiltonianWithBasis[TruncatedBasis[_L3, MomentumBasis[_L0]], TruncatedBasis[_L4, MomentumBasis[_L1]], ExplicitBasis[_L5, PositionBasis[_L2]]]
        Hamiltonian in the specified basis
    """
    potential = get_interpolated_potential_relaxed(shape)
    potential["points"] = 0.5 * (
        potential["points"] + potential["points"].swapaxes(0, 1)
    )
    config: SHOBasisConfig = {
        "sho_omega": 111119431700988.45,
        "mass": 1.6735575e-27,
        "x_origin": np.array([0, 0, -1.8866087481825024e-10]),
    }
    return sho_subtracted_basis.total_surface_hamiltonian(
        potential, config, bloch_phase, resolution
    )
