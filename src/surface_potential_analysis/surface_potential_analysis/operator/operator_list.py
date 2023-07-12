from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar

from surface_potential_analysis.basis.basis import (
    Basis,
)

if TYPE_CHECKING:
    import numpy as np

    from surface_potential_analysis._types import SingleFlatIndexLike

    from .operator import (
        DiagonalOperator,
        Operator,
    )
_B1Inv = TypeVar("_B1Inv", bound=Basis)
_B0Inv = TypeVar("_B0Inv", bound=Basis)
_L0Inv = TypeVar("_L0Inv", bound=int)


class OperatorList(TypedDict, Generic[_B0Inv, _B1Inv, _L0Inv]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    basis: _B0Inv
    dual_basis: _B1Inv
    arrays: np.ndarray[tuple[_L0Inv, int, int], np.dtype[np.complex_]]
    """A list of state vectors"""


def get_operator(
    operator_list: OperatorList[_B0Inv, _B1Inv, _L0Inv], idx: SingleFlatIndexLike
) -> Operator[_B0Inv, _B1Inv]:
    """
    Get a single state vector from a list of states.

    Parameters
    ----------
    list : EigenstateList[_B0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    Eigenstate[_B0Inv]
    """
    return {
        "basis": operator_list["basis"],
        "dual_basis": operator_list["dual_basis"],
        "array": operator_list["arrays"][idx],
    }


class DiagonalOperatorList(TypedDict, Generic[_B0Inv, _B1Inv, _L0Inv]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    basis: _B0Inv
    dual_basis: _B1Inv
    vectors: np.ndarray[tuple[_L0Inv, int], np.dtype[np.complex_]]
    """A list of state vectors"""


def get_diagonal_operator(
    operator_list: DiagonalOperatorList[_B0Inv, _B1Inv, _L0Inv],
    idx: SingleFlatIndexLike,
) -> DiagonalOperator[_B0Inv, _B1Inv]:
    """
    Get a single state vector from a list of states.

    Parameters
    ----------
    list : EigenstateList[_B0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    Eigenstate[_B0Inv]
    """
    return {
        "basis": operator_list["basis"],
        "dual_basis": operator_list["dual_basis"],
        "vector": operator_list["vectors"][idx],
    }
