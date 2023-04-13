"""Tools for generating a surface hamiltonian in the momentum basis."""


from typing import Literal, TypeVar

import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.basis_config import BasisConfigUtil, MomentumBasisConfig
from surface_potential_analysis.hamiltonian import (
    MomentumBasisHamiltonian,
    PositionBasisStackedHamiltonian,
    add_hamiltonian,
    convert_stacked_hamiltonian_to_momentum_basis,
    flatten_hamiltonian,
)
from surface_potential_analysis.potential import Potential
from surface_potential_analysis.util import timed

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
    potential_in_momentum = flatten_hamiltonian(
        convert_stacked_hamiltonian_to_momentum_basis(potential_hamiltonian)
    )

    kinetic = hamiltonian_from_mass(potential_in_momentum["basis"], mass, bloch_phase)
    return add_hamiltonian(potential_in_momentum, kinetic)
