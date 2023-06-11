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

    from .operator import (
        Operator,
    )

    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
    _B1Inv = TypeVar("_B1Inv", bound=Basis[Any])
    _B2Inv = TypeVar("_B2Inv", bound=Basis[Any])
    _B3Inv = TypeVar("_B3Inv", bound=Basis[Any])

    _B1d0Inv = TypeVar("_B1d0Inv", bound=Basis1d[Any])
    _B1d1Inv = TypeVar("_B1d1Inv", bound=Basis1d[Any])
    _B1d2Inv = TypeVar("_B1d2Inv", bound=Basis1d[Any])
    _B1d3Inv = TypeVar("_B1d3Inv", bound=Basis1d[Any])
    _B2d0Inv = TypeVar("_B2d0Inv", bound=Basis2d[Any, Any])
    _B2d1Inv = TypeVar("_B2d1Inv", bound=Basis2d[Any, Any])
    _B2d2Inv = TypeVar("_B2d2Inv", bound=Basis2d[Any, Any])
    _B2d3Inv = TypeVar("_B2d3Inv", bound=Basis2d[Any, Any])
    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])
    _B3d1Inv = TypeVar("_B3d1Inv", bound=Basis3d[Any, Any, Any])
    _B3d2Inv = TypeVar("_B3d2Inv", bound=Basis3d[Any, Any, Any])
    _B3d3Inv = TypeVar("_B3d3Inv", bound=Basis3d[Any, Any, Any])


@overload
def convert_operator_to_basis(
    operator: Operator[_B1d0Inv, _B1d1Inv], basis: _B1d2Inv, dual_basis: _B1d3Inv
) -> Operator[_B1d2Inv, _B1d3Inv]:
    ...


@overload
def convert_operator_to_basis(
    operator: Operator[_B2d0Inv, _B2d1Inv], basis: _B2d2Inv, dual_basis: _B2d3Inv
) -> Operator[_B2d2Inv, _B2d3Inv]:
    ...


@overload
def convert_operator_to_basis(
    operator: Operator[_B3d0Inv, _B3d1Inv], basis: _B3d2Inv, dual_basis: _B3d3Inv
) -> Operator[_B3d2Inv, _B3d3Inv]:
    ...


@overload
def convert_operator_to_basis(
    operator: Operator[_B0Inv, _B1Inv], basis: _B2Inv, dual_basis: _B3Inv
) -> Operator[_B2Inv, _B3Inv]:
    ...


def convert_operator_to_basis(
    operator: Operator[_B0Inv, _B1Inv], basis: _B2Inv, dual_basis: _B3Inv
) -> Operator[_B2Inv, _B3Inv]:
    """
    Given an operator, convert it to the given basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
    basis : _B3d1Inv

    Returns
    -------
    Eigenstate[_B3d1Inv]
    """
    converted = convert_matrix(
        operator["array"], operator["basis"], basis, operator["dual_basis"], dual_basis
    )
    return {"basis": basis, "dual_basis": dual_basis, "array": converted}
