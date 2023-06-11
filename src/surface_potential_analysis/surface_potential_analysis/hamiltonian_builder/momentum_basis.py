from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.axis.conversion import axis_as_fundamental_momentum_axis
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_position_basis,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.operator.conversion import (
    convert_operator_to_basis,
)
from surface_potential_analysis.operator.operator import (
    SingleBasisOperator,
    add_operator,
)
from surface_potential_analysis.potential.conversion import convert_potential_to_basis
from surface_potential_analysis.util.decorators import timed

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        Basis,
    )
    from surface_potential_analysis.potential.potential import Potential

    _L0 = TypeVar("_L0", bound=int)
    _L1 = TypeVar("_L1", bound=int)
    _L2 = TypeVar("_L2", bound=int)
    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])


def hamiltonian_from_potential(
    potential: Potential[_B0Inv],
) -> SingleBasisOperator[_B0Inv]:
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
        potential, basis_as_fundamental_position_basis(potential["basis"])
    )

    return convert_operator_to_basis(
        {
            "basis": converted["basis"],
            "dual_basis": converted["basis"],
            "array": np.diag(converted["vector"]),
        },
        potential["basis"],
        potential["basis"],
    )


def hamiltonian_from_mass(
    basis: _B0Inv,
    mass: float,
    bloch_fraction: np.ndarray[tuple[_L0], np.dtype[np.float_]] | None = None,
) -> SingleBasisOperator[_B0Inv]:
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
    bloch_fraction = np.zeros(len(basis)) if bloch_fraction is None else bloch_fraction
    util = BasisUtil(basis)

    bloch_phase = np.tensordot(util.fundamental_dk, bloch_fraction, axes=(0, 0))
    k_points = util.fundamental_k_points + bloch_phase[:, np.newaxis]
    energy = np.sum(np.square(hbar * k_points) / (2 * mass), axis=0)
    momentum_basis = tuple(axis_as_fundamental_momentum_axis(ax) for ax in basis)

    hamiltonian: SingleBasisOperator[Any] = {
        "basis": momentum_basis,
        "dual_basis": momentum_basis,
        "array": np.diag(energy),
    }
    return convert_operator_to_basis(hamiltonian, basis, basis)


@timed
def total_surface_hamiltonian(
    potential: Potential[_B0Inv],
    mass: float,
    bloch_fraction: np.ndarray[tuple[_L0], np.dtype[np.float_]] | None = None,
) -> SingleBasisOperator[_B0Inv]:
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
        potential["basis"], mass, bloch_fraction
    )

    return add_operator(kinetic_hamiltonian, potential_hamiltonian)
