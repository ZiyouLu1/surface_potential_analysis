from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.axis.stacked_axis import StackedBasis
from surface_potential_analysis.axis.util import (
    BasisUtil,
)
from surface_potential_analysis.operator.conversion import (
    add_operator,
    convert_operator_to_basis,
)
from surface_potential_analysis.potential.conversion import convert_potential_to_basis
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.axis.stacked_axis import StackedBasisLike
    from surface_potential_analysis.operator.operator import (
        SingleBasisOperator,
    )
    from surface_potential_analysis.potential.potential import Potential

    _L0 = TypeVar("_L0", bound=int)
    _SB0 = TypeVar("_SB0", bound=StackedBasisLike[*tuple[Any, ...]])


def hamiltonian_from_potential(
    potential: Potential[_SB0],
) -> SingleBasisOperator[_SB0]:
    """
    Given a potential in some basis get the hamiltonian in the same basis.

    Parameters
    ----------
    potential : Potential[_B0Inv]

    Returns
    -------
    Hamiltonian[_B0Inv]
    """
    converted = convert_potential_to_basis(
        potential, stacked_basis_as_fundamental_position_basis(potential["basis"])
    )

    return convert_operator_to_basis(
        {
            "basis": StackedBasis(converted["basis"], converted["basis"]),
            "data": np.diag(converted["data"]).reshape(-1),
        },
        StackedBasis(potential["basis"], potential["basis"]),
    )


def hamiltonian_from_mass(
    basis: _SB0,
    mass: float,
    bloch_fraction: np.ndarray[tuple[_L0], np.dtype[np.float_]] | None = None,
) -> SingleBasisOperator[_SB0]:
    """
    Given a mass and a basis calculate the kinetic part of the Hamiltonian.

    Parameters
    ----------
    basis : _B0Inv
    mass : float
    bloch_fraction : np.ndarray[tuple[int], np.dtype[np.float_]] | None, optional
        bloch phase, by default None

    Returns
    -------
    Hamiltonian[_B0Inv]
    """
    bloch_fraction = np.zeros(basis.ndim) if bloch_fraction is None else bloch_fraction
    util = BasisUtil(basis)

    bloch_phase = np.tensordot(util.fundamental_dk_stacked, bloch_fraction, axes=(0, 0))
    k_points = util.fundamental_stacked_k_points + bloch_phase[:, np.newaxis]
    energy = np.sum(np.square(hbar * k_points) / (2 * mass), axis=0, dtype=np.complex_)
    momentum_basis = stacked_basis_as_fundamental_momentum_basis(basis)

    hamiltonian: SingleBasisOperator[Any] = {
        "basis": StackedBasis(momentum_basis, momentum_basis),
        "data": np.diag(energy).reshape(-1),
    }
    return convert_operator_to_basis(hamiltonian, StackedBasis(basis, basis))


def total_surface_hamiltonian(
    potential: Potential[_SB0],
    mass: float,
    bloch_fraction: np.ndarray[tuple[_L0], np.dtype[np.float_]] | None = None,
) -> SingleBasisOperator[_SB0]:
    """
    Calculate the total hamiltonian in momentum basis for a given potential and mass.

    Parameters
    ----------
    potential : Potential[_L0, _L1, _L2]
    mass : float
    bloch_fraction : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]

    Returns
    -------
    MomentumBasisHamiltonian[_L0, _L1, _L2]
    """
    potential_hamiltonian = hamiltonian_from_potential(potential)
    kinetic_hamiltonian = hamiltonian_from_mass(
        potential_hamiltonian["basis"][0], mass, bloch_fraction
    )

    return add_operator(kinetic_hamiltonian, potential_hamiltonian)
