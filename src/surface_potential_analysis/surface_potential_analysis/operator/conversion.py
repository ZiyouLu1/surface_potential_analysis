from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.axis.axis_like import convert_matrix

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis_like import BasisLike
    from surface_potential_analysis.axis.stacked_axis import StackedBasisLike
    from surface_potential_analysis.operator.operator import (
        Operator,
    )

    _B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])
    _B1Inv = TypeVar("_B1Inv", bound=BasisLike[Any, Any])
    _B2Inv = TypeVar("_B2Inv", bound=BasisLike[Any, Any])
    _B3Inv = TypeVar("_B3Inv", bound=BasisLike[Any, Any])


def convert_operator_to_basis(
    operator: Operator[_B0Inv, _B1Inv], basis: StackedBasisLike[_B2Inv, _B3Inv]
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
        operator["data"].reshape(operator["basis"].shape),
        operator["basis"][0],
        basis[0],
        operator["basis"][1],
        basis[1],
    )
    return {"basis": basis, "data": converted.reshape(-1)}


def add_operator(
    a: Operator[_B0Inv, _B1Inv], b: Operator[_B0Inv, _B1Inv]
) -> Operator[_B0Inv, _B1Inv]:
    """
    Add together two operators.

    Parameters
    ----------
    a : Operator[_B0Inv]
    b : Operator[_B0Inv]

    Returns
    -------
    Operator[_B0Inv]
    """
    return {"basis": a["basis"], "data": a["data"] + b["data"]}
