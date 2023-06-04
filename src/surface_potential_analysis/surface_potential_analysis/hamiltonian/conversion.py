from __future__ import annotations

from typing import Any, TypeVar, overload

import numpy as np

from surface_potential_analysis.basis.basis import (
    Basis,
    Basis1d,
    Basis2d,
    Basis3d,
)
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_momentum_basis,
    basis_as_fundamental_position_basis,
    convert_matrix,
)

from .hamiltonian import (
    FundamentalMomentumBasisHamiltonian3d,
    FundamentalMomentumBasisStackedHamiltonian3d,
    FundamentalPositionBasisHamiltonian3d,
    FundamentalPositionBasisStackedHamiltonian3d,
    Hamiltonian,
    flatten_hamiltonian,
    stack_hamiltonian,
)

_B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
_B1Inv = TypeVar("_B1Inv", bound=Basis[Any])

_B1d0Inv = TypeVar("_B1d0Inv", bound=Basis1d[Any])
_B1d1Inv = TypeVar("_B1d1Inv", bound=Basis1d[Any])
_B2d0Inv = TypeVar("_B2d0Inv", bound=Basis2d[Any, Any])
_B2d1Inv = TypeVar("_B2d1Inv", bound=Basis2d[Any, Any])
_B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])
_B3d1Inv = TypeVar("_B3d1Inv", bound=Basis3d[Any, Any, Any])

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


@overload
def convert_hamiltonian_to_basis(
    hamiltonian: Hamiltonian[_B1d0Inv], basis: _B1d1Inv
) -> Hamiltonian[_B1d1Inv]:
    ...


@overload
def convert_hamiltonian_to_basis(
    hamiltonian: Hamiltonian[_B2d0Inv], basis: _B2d1Inv
) -> Hamiltonian[_B2d1Inv]:
    ...


@overload
def convert_hamiltonian_to_basis(
    hamiltonian: Hamiltonian[_B3d0Inv], basis: _B3d1Inv
) -> Hamiltonian[_B3d1Inv]:
    ...


@overload
def convert_hamiltonian_to_basis(
    hamiltonian: Hamiltonian[_B0Inv], basis: _B1Inv
) -> Hamiltonian[_B1Inv]:
    ...


def convert_hamiltonian_to_basis(
    hamiltonian: Hamiltonian[_B0Inv], basis: _B1Inv
) -> Hamiltonian[_B1Inv]:
    """
    Given a hamiltonian, convert it to the given basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
    basis : _B3d1Inv

    Returns
    -------
    Eigenstate[_B3d1Inv]
    """
    converted = convert_matrix(
        hamiltonian["array"].astype(np.complex_), hamiltonian["basis"], basis
    )
    return {"basis": basis, "array": converted}


def _convert_stacked_hamiltonian_to_momentum_basis(
    hamiltonian: FundamentalPositionBasisStackedHamiltonian3d[_L0Inv, _L1Inv, _L2Inv]
) -> FundamentalMomentumBasisStackedHamiltonian3d[_L0Inv, _L1Inv, _L2Inv]:
    transformed = np.fft.ifftn(
        np.fft.fftn(hamiltonian["array"], axes=(0, 1, 2), norm="ortho"),
        axes=(3, 4, 5),
        norm="ortho",
    )
    basis = basis_as_fundamental_momentum_basis(hamiltonian["basis"])
    return {"basis": basis, "array": transformed}


def convert_hamiltonian_to_momentum_basis(
    hamiltonian: FundamentalPositionBasisHamiltonian3d[_L0Inv, _L1Inv, _L2Inv]
) -> FundamentalMomentumBasisHamiltonian3d[_L0Inv, _L1Inv, _L2Inv]:
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
    hamiltonian: FundamentalMomentumBasisStackedHamiltonian3d[_L0Inv, _L1Inv, _L2Inv]
) -> FundamentalPositionBasisStackedHamiltonian3d[_L0Inv, _L1Inv, _L2Inv]:
    # TODO: which way round
    transformed = np.fft.fftn(
        np.fft.ifftn(hamiltonian["array"], axes=(0, 1, 2), norm="ortho"),
        axes=(3, 4, 5),
        norm="ortho",
    )
    basis = basis_as_fundamental_position_basis(hamiltonian["basis"])
    return {"basis": basis, "array": transformed}


def convert_hamiltonian_to_position_basis(
    hamiltonian: FundamentalMomentumBasisHamiltonian3d[_L0Inv, _L1Inv, _L2Inv]
) -> FundamentalPositionBasisHamiltonian3d[_L0Inv, _L1Inv, _L2Inv]:
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
