from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.basis_config.basis_config import (
    BasisConfigUtil,
    MomentumBasisConfig,
)
from surface_potential_analysis.hamiltonian.conversion import (
    convert_hamiltonian_to_momentum_basis,
)
from surface_potential_analysis.hamiltonian.hamiltonian import (
    MomentumBasisHamiltonian,
    PositionBasisStackedHamiltonian,
    add_hamiltonian,
    flatten_hamiltonian,
)
from surface_potential_analysis.util import timed

if TYPE_CHECKING:
    from surface_potential_analysis.potential import Potential

_L0 = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
_L2 = TypeVar("_L2", bound=int)


def hamiltonian_from_potential(
    potential: Potential[_L0, _L1, _L2],
) -> PositionBasisStackedHamiltonian[_L0, _L1, _L2]:
    """Given a potential in position basis [ix0, ix1, ix2], get the hamiltonian in stacked form.

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
    shape = (*potential["points"].shape, *potential["points"].shape)
    array = np.diagflat(potential["points"]).reshape(shape)
    return {"basis": potential["basis"], "array": array}


def hamiltonian_from_mass(
    basis: MomentumBasisConfig[_L0, _L1, _L2],
    mass: float,
    bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]] | None = None,
) -> MomentumBasisHamiltonian[_L0, _L1, _L2]:
    """
    Calculate the kinetic hamiltonain for a particle with given mass.

    Parameters
    ----------
    basis : MomentumBasisConfig[_L0, _L1, _L2]
        basis to calculate the potential for
    mass : float
        mass of the particle
    bloch_phase : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]] | None, optional
        bloch phase, by default [0,0,0]

    Returns
    -------
    MomentumBasisHamiltonian[_L0, _L1, _L2]
        _description_
    """
    bloch_phase = np.array([0.0, 0.0, 0.0]) if bloch_phase is None else bloch_phase
    util = BasisConfigUtil(basis)

    k0_coords, k1_coords, k2_coords = util.fundamental_nk_points
    k0_coords += bloch_phase[0]
    k1_coords += bloch_phase[1]
    k2_coords += bloch_phase[2]

    dk0 = util.dk0
    dk1 = util.dk1
    dk2 = util.dk2

    kx_points = dk0[0] * k0_coords + dk1[0] * k1_coords + dk2[0] * k2_coords
    x_energy = (hbar * (kx_points)) ** 2 / (2 * mass)
    ky_points = dk0[1] * k0_coords + dk1[1] * k1_coords + dk2[1] * k2_coords
    y_energy = (hbar * ky_points) ** 2 / (2 * mass)
    kz_points = dk0[2] * k0_coords + dk1[2] * k1_coords + dk2[2] * k2_coords
    z_energy = (hbar * kz_points) ** 2 / (2 * mass)

    energy = x_energy + y_energy + z_energy
    return {"basis": basis, "array": np.diag(energy)}


@timed
def total_surface_hamiltonian(
    potential: Potential[_L0, _L1, _L2],
    mass: float,
    bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
) -> MomentumBasisHamiltonian[_L0, _L1, _L2]:
    """
    Calculate the total hamiltonian in momentum basis for a given potential and mass.

    Parameters
    ----------
    potential : Potential[_L0, _L1, _L2]
    mass : float
    bloch_phase : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]

    Returns
    -------
    MomentumBasisHamiltonian[_L0, _L1, _L2]
    """
    potential_hamiltonian = hamiltonian_from_potential(potential)
    potential_in_momentum = convert_hamiltonian_to_momentum_basis(
        flatten_hamiltonian(potential_hamiltonian)
    )

    kinetic = hamiltonian_from_mass(potential_in_momentum["basis"], mass, bloch_phase)
    return add_hamiltonian(potential_in_momentum, kinetic)
