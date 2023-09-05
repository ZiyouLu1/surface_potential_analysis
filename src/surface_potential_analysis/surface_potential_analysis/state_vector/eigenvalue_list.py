from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import Basis
from surface_potential_analysis.basis.util import BasisUtil

if TYPE_CHECKING:
    from surface_potential_analysis._types import SingleFlatIndexLike

_B0Inv = TypeVar("_B0Inv", bound=Basis)


class EigenvalueList(TypedDict, Generic[_B0Inv]):
    """Represents a list of eigenvalues."""

    list_basis: _B0Inv
    eigenvalues: np.ndarray[tuple[int], np.dtype[np.complex_]]
    """A list of eigenvalues"""


def get_eigenvalue(
    eigenvalue_list: EigenvalueList[_B0Inv], idx: SingleFlatIndexLike
) -> np.complex_:
    """
    Get a single eigenvalue from the list.

    Parameters
    ----------
    eigenvalue_list : EigenvalueList[_L0Inv]
    idx : SingleFlatIndexLike

    Returns
    -------
    np.complex_
    """
    return eigenvalue_list["eigenvalues"][idx]  # type: ignore[return-value]


def average_eigenvalues(
    eigenvalues: EigenvalueList[_B0Inv],
    axis: tuple[int, ...] | None = None,
    *,
    weights: np.ndarray[tuple[int], np.dtype[np.float_]] | None = None,
) -> EigenvalueList[Any]:
    """
    Average eigenvalues over the given axis.

    Parameters
    ----------
    eigenvalues : EigenvalueList[_B0Inv]
    axis : tuple[int, ...] | None, optional
        axis, by default None
    weights : np.ndarray[tuple[int], np.dtype[np.float_]] | None, optional
        weights, by default None

    Returns
    -------
    EigenvalueList[Any]
    """
    axis = tuple(range(len(eigenvalues["list_basis"]))) if axis is None else axis
    util = BasisUtil(eigenvalues["list_basis"])
    basis = tuple(b for (i, b) in enumerate(eigenvalues["list_basis"]) if i not in axis)
    return {
        "list_basis": basis,
        "eigenvalues": np.average(
            eigenvalues["eigenvalues"].reshape(*util.shape),
            axis=tuple(ax for ax in axis),
            weights=weights,
        ).reshape(-1),
    }
