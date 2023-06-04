from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.axis.conversion import axis_as_fundamental_momentum_axis
from surface_potential_analysis.basis.util import (
    Basis3dUtil,
    BasisUtil,
)
from surface_potential_analysis.hamiltonian.conversion import (
    convert_hamiltonian_to_basis,
    convert_hamiltonian_to_momentum_basis,
)
from surface_potential_analysis.hamiltonian.hamiltonian import (
    FundamentalMomentumBasisHamiltonian3d,
    FundamentalPositionBasisStackedHamiltonian3d,
    Hamiltonian,
    add_hamiltonian,
    flatten_hamiltonian,
)
from surface_potential_analysis.util.decorators import timed

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        Basis,
        FundamentalMomentumBasis3d,
    )
    from surface_potential_analysis.potential import (
        FundamentalPositionBasisPotential3d,
    )
    from surface_potential_analysis.potential.potential import Potential

    _L0 = TypeVar("_L0", bound=int)
    _L1 = TypeVar("_L1", bound=int)
    _L2 = TypeVar("_L2", bound=int)
    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])


def hamiltonian_from_potential(potential: Potential[_B0Inv]) -> Hamiltonian[_B0Inv]:
    """
    Given a potential in some basis get the hamiltonian in the same basis.

    Parameters
    ----------
    potential : Potential[_B0Inv]

    Returns
    -------
    Hamiltonian[_B0Inv]
    """
    return {"basis": potential["basis"], "array": np.diag(potential["vector"])}


def hamiltonian_from_position_basis_potential_3d_stacked(
    potential: FundamentalPositionBasisPotential3d[_L0, _L1, _L2],
) -> FundamentalPositionBasisStackedHamiltonian3d[_L0, _L1, _L2]:
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
    shape = BasisUtil(potential["basis"]).shape
    array = np.diag(potential["vector"]).reshape(*shape, *shape)
    return {"basis": potential["basis"], "array": array}


def hamiltonian_from_mass(
    basis: _B0Inv,
    mass: float,
    bloch_phase: np.ndarray[tuple[_L0], np.dtype[np.float_]] | None = None,
) -> Hamiltonian[_B0Inv]:
    """
    Given a mass and a basis calculate the kinetic part of the Hamiltonian.

    Parameters
    ----------
    basis : _B0Inv
    mass : float
    bloch_phase : np.ndarray[tuple[int], np.dtype[np.float_]] | None, optional
        bloch phase, by default None

    Returns
    -------
    Hamiltonian[_B0Inv]
    """
    bloch_phase = np.array([0.0 for _ in basis]) if bloch_phase is None else bloch_phase
    util = BasisUtil(basis)
    k_points = util.fundamental_k_points + bloch_phase[:, np.newaxis]
    energy = np.sum(np.square(hbar * k_points) / (2 * mass), axis=0)
    momentum_basis = tuple(axis_as_fundamental_momentum_axis(ax) for ax in basis)
    momentum_hamiltonian: Hamiltonian[Any] = {
        "basis": momentum_basis,
        "array": np.diag(energy),
    }
    return convert_hamiltonian_to_basis(momentum_hamiltonian, basis)


def fundamental_hamiltonian_3d_from_mass(
    basis: FundamentalMomentumBasis3d[_L0, _L1, _L2],
    mass: float,
    bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]] | None = None,
) -> FundamentalMomentumBasisHamiltonian3d[_L0, _L1, _L2]:
    """
    Calculate the kinetic hamiltonain for a particle with given mass.

    Parameters
    ----------
    basis : MomentumBasis3d[_L0, _L1, _L2]
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
    util = Basis3dUtil(basis)

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
    potential: FundamentalPositionBasisPotential3d[_L0, _L1, _L2],
    mass: float,
    bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
) -> FundamentalMomentumBasisHamiltonian3d[_L0, _L1, _L2]:
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
    potential_hamiltonian = hamiltonian_from_position_basis_potential_3d_stacked(
        potential
    )
    potential_in_momentum = convert_hamiltonian_to_momentum_basis(
        flatten_hamiltonian(potential_hamiltonian)
    )

    kinetic = fundamental_hamiltonian_3d_from_mass(
        potential_in_momentum["basis"], mass, bloch_phase
    )
    return add_hamiltonian(potential_in_momentum, kinetic)
