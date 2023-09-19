from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis_like import BasisLike
from surface_potential_analysis.axis.stacked_axis import StackedBasis, StackedBasisLike
from surface_potential_analysis.axis.util import BasisUtil

if TYPE_CHECKING:
    from surface_potential_analysis.types import SingleFlatIndexLike

_B0_co = TypeVar("_B0_co", bound=BasisLike[Any, Any], covariant=True)


class EigenvalueList(TypedDict, Generic[_B0_co]):
    """Represents a list of eigenvalues."""

    basis: _B0_co
    data: np.ndarray[tuple[int], np.dtype[np.complex_]]
    """A list of eigenvalues"""


def get_eigenvalue(
    eigenvalue_list: EigenvalueList[BasisLike[Any, Any]], idx: SingleFlatIndexLike
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
    return eigenvalue_list["data"][idx]


def average_eigenvalues(
    eigenvalues: EigenvalueList[StackedBasisLike[*tuple[Any, ...]]],
    axis: tuple[int, ...] | None = None,
    *,
    weights: np.ndarray[tuple[int], np.dtype[np.float_]] | None = None,
) -> EigenvalueList[StackedBasis[*tuple[Any, ...]]]:
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
    axis = tuple(range(eigenvalues["basis"].ndim)) if axis is None else axis
    util = BasisUtil(eigenvalues["basis"])
    basis = tuple(b for (i, b) in enumerate(eigenvalues["basis"]) if i not in axis)
    return {
        "basis": StackedBasis(*basis),
        "data": np.average(
            eigenvalues["data"].reshape(*util.shape),
            axis=tuple(ax for ax in axis),
            weights=weights,
        ).reshape(-1),
    }
