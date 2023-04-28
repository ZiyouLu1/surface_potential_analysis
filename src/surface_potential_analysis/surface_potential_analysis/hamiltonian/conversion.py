from typing import Any, TypeVar

import numpy as np

from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    MomentumBasisConfigUtil,
    PositionBasisConfigUtil,
)
from surface_potential_analysis.basis_config.conversion import convert_matrix

from .hamiltonian import (
    Hamiltonian,
    MomentumBasisHamiltonian,
    MomentumBasisStackedHamiltonian,
    PositionBasisHamiltonian,
    PositionBasisStackedHamiltonian,
    flatten_hamiltonian,
    stack_hamiltonian,
)

_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])
_BC1Inv = TypeVar("_BC1Inv", bound=BasisConfig[Any, Any, Any])

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def convert_hamiltonian_to_basis(
    hamiltonian: Hamiltonian[_BC0Inv], basis: _BC1Inv
) -> Hamiltonian[_BC1Inv]:
    """
    Given an eigenstate, calculate the vector in the given basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    basis : _BC1Inv

    Returns
    -------
    Eigenstate[_BC1Inv]
    """
    converted = convert_matrix(
        hamiltonian["array"].astype(np.complex_), hamiltonian["basis"], basis
    )
    return {"basis": basis, "array": converted}


def _convert_stacked_hamiltonian_to_momentum_basis(
    hamiltonian: PositionBasisStackedHamiltonian[_L0Inv, _L1Inv, _L2Inv]
) -> MomentumBasisStackedHamiltonian[_L0Inv, _L1Inv, _L2Inv]:
    transformed = np.fft.ifftn(
        np.fft.fftn(hamiltonian["array"], axes=(0, 1, 2), norm="ortho"),
        axes=(3, 4, 5),
        norm="ortho",
    )
    util = PositionBasisConfigUtil(hamiltonian["basis"])
    return {
        "basis": util.get_reciprocal_basis(),
        "array": transformed,
    }


def convert_hamiltonian_to_momentum_basis(
    hamiltonian: PositionBasisHamiltonian[_L0Inv, _L1Inv, _L2Inv]
) -> MomentumBasisHamiltonian[_L0Inv, _L1Inv, _L2Inv]:
    """
    Convert a hamiltonian from position to momentum basis.

    Parameters
    ----------
    hamiltonian : PositionBasisHamiltonian[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    MomentumBasisHamiltonian[_L0Inv, _L1Inv, _L2Inv]
    """
    stacked = stack_hamiltonian(hamiltonian)
    converted = _convert_stacked_hamiltonian_to_momentum_basis(stacked)
    return flatten_hamiltonian(converted)


def _convert_stacked_hamiltonian_to_position_basis(
    hamiltonian: MomentumBasisStackedHamiltonian[_L0Inv, _L1Inv, _L2Inv]
) -> PositionBasisStackedHamiltonian[_L0Inv, _L1Inv, _L2Inv]:
    # TODO: which way round
    transformed = np.fft.fftn(
        np.fft.ifftn(hamiltonian["array"], axes=(0, 1, 2), norm="ortho"),
        axes=(3, 4, 5),
        norm="ortho",
    )
    util = MomentumBasisConfigUtil(hamiltonian["basis"])
    return {
        "basis": util.get_reciprocal_basis(),
        "array": transformed,
    }


def convert_hamiltonian_to_position_basis(
    hamiltonian: MomentumBasisHamiltonian[_L0Inv, _L1Inv, _L2Inv]
) -> PositionBasisHamiltonian[_L0Inv, _L1Inv, _L2Inv]:
    """
    Convert a hamiltonian from momentum to position basis.

    Parameters
    ----------
    hamiltonian : MomentumBasisHamiltonian[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    PositionBasisHamiltonian[_L0Inv, _L1Inv, _L2Inv]
    """
    stacked = stack_hamiltonian(hamiltonian)
    converted = _convert_stacked_hamiltonian_to_position_basis(stacked)
    return flatten_hamiltonian(converted)
