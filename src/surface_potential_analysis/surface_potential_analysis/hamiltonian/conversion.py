from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload

from surface_potential_analysis.basis.conversion import (
    convert_matrix,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        Basis,
        Basis1d,
        Basis2d,
        Basis3d,
    )

    from .hamiltonian import (
        Hamiltonian,
    )

    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
    _B1Inv = TypeVar("_B1Inv", bound=Basis[Any])

    _B1d0Inv = TypeVar("_B1d0Inv", bound=Basis1d[Any])
    _B1d1Inv = TypeVar("_B1d1Inv", bound=Basis1d[Any])
    _B2d0Inv = TypeVar("_B2d0Inv", bound=Basis2d[Any, Any])
    _B2d1Inv = TypeVar("_B2d1Inv", bound=Basis2d[Any, Any])
    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])
    _B3d1Inv = TypeVar("_B3d1Inv", bound=Basis3d[Any, Any, Any])


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
    converted = convert_matrix(hamiltonian["array"], hamiltonian["basis"], basis)
    return {"basis": basis, "array": converted}
