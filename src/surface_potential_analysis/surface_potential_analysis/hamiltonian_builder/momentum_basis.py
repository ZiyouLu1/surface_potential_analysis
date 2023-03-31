from typing import Any, Literal, TypeVar

import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.basis_config import (
    MomentumBasisConfig,
    MomentumBasisConfigUtil,
)
from surface_potential_analysis.hamiltonian import (
    MomentumBasisHamiltonian,
    MomentumBasisStackedHamiltonian,
    PositionBasisStackedHamiltonian,
    add_hamiltonian_stacked,
    convert_stacked_hamiltonian_to_momentum_basis,
    flatten_hamiltonian,
)
from surface_potential_analysis.potential import Potential

_L0 = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
_L2 = TypeVar("_L2", bound=int)

_DT0 = TypeVar("_DT0", bound=np.dtype[Any])


def _diag_along_axis(
    points: np.ndarray[Any, _DT0], axis: int = -1
) -> np.ndarray[Any, _DT0]:
    return np.apply_along_axis(np.diag, axis, points)  # type:ignore


def hamiltonian_from_potential(
    potential: Potential[_L0, _L1, _L2],
) -> PositionBasisStackedHamiltonian[_L0, _L1, _L2]:
    """
    Given a potential in position basis [ix0, ix1, ix2],
    get the hamiltonian in stacked form.

    This is just a matrix with the potential along the diagonals

    Parameters
    ----------
    potential : NDArray
        The potential in position basis [ix0, ix1, ix2]

    Returns
    -------
    HamiltonianStacked[PositionBasis, PositionBasis, PositionBasis]
        The hamiltonian in stacked form
    """

    hamiltonian = _diag_along_axis(
        _diag_along_axis(_diag_along_axis(potential["points"], axis=2), axis=1),
        axis=0,
    )
    hamiltonian = np.moveaxis(hamiltonian, 1, -1)
    hamiltonian = np.moveaxis(hamiltonian, 2, -1)
    hamiltonian = np.moveaxis(hamiltonian, 3, -1)

    return {"basis": potential["basis"], "array": hamiltonian}


def hamiltonian_from_mass(
    basis: MomentumBasisConfig[_L0, _L1, _L2],
    mass: float,
    bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]] | None = None,
) -> MomentumBasisStackedHamiltonian[_L0, _L1, _L2]:
    bloch_phase = np.array([0.0, 0.0, 0.0]) if bloch_phase is None else bloch_phase
    util = MomentumBasisConfigUtil(basis)

    kx0_coords, kx1_coords, kx2_coords = util.fundamental_nk_points
    kx0_coords += bloch_phase[0]
    kx1_coords += bloch_phase[1]
    kx2_coords += bloch_phase[2]

    dk0 = util.dk0
    dk1 = util.dk1
    dk2 = util.dk2

    kx_points = dk0[0] * kx0_coords + dk1[0] * kx1_coords + dk2[0] * kx2_coords
    x_energy = (hbar * (kx_points)) ** 2 / (2 * mass)
    ky_points = dk0[1] * kx0_coords + dk1[1] * kx1_coords + dk2[1] * kx2_coords
    y_energy = (hbar * ky_points) ** 2 / (2 * mass)
    kz_points = dk0[2] * kx0_coords + dk1[2] * kx1_coords + dk2[2] * kx2_coords
    z_energy = (hbar * kz_points) ** 2 / (2 * mass)

    energy = x_energy + y_energy + z_energy
    return {"basis": basis, "array": np.diag(energy)}


def total_surface_hamiltonian_stacked(
    potential: Potential[_L0, _L1, _L2],
    mass: float,
    bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
) -> MomentumBasisStackedHamiltonian[_L0, _L1, _L2]:
    potential_hamiltonian = hamiltonian_from_potential(potential)
    potential_in_momentum = convert_stacked_hamiltonian_to_momentum_basis(
        potential_hamiltonian
    )

    kinetic = hamiltonian_from_mass(potential_in_momentum["basis"], mass, bloch_phase)

    return add_hamiltonian_stacked(potential_in_momentum, kinetic)


def total_surface_hamiltonian(
    potential: Potential[_L0, _L1, _L2],
    mass: float,
    bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
) -> MomentumBasisHamiltonian[_L0, _L1, _L2]:
    stacked = total_surface_hamiltonian_stacked(potential, mass, bloch_phase)
    return flatten_hamiltonian(stacked)
