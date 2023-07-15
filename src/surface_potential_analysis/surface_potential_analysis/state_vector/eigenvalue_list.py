from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar

if TYPE_CHECKING:
    import numpy as np

    from surface_potential_analysis._types import SingleFlatIndexLike

_L0Inv = TypeVar("_L0Inv", bound=int)


class EigenvalueList(TypedDict, Generic[_L0Inv]):
    """Represents a list of eigenvalues."""

    eigenvalues: np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
    """A list of eigenvalues"""


def get_eigenvalue(
    eigenvalue_list: EigenvalueList[_L0Inv], idx: SingleFlatIndexLike
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
