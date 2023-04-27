from typing import Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.conversion import get_basis_conversion_matrix
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
)

_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])
_BC1Inv = TypeVar("_BC1Inv", bound=BasisConfig[Any, Any, Any])


def convert_vector(
    vector: np.ndarray[tuple[int], np.dtype[np.complex_]],
    initial_basis: _BC0Inv,
    final_basis: _BC1Inv,
) -> np.ndarray[tuple[int], np.dtype[np.complex_]]:
    """
    Convert a vector, expressed in terms of the given basis from_config in the basis to_config.

    Parameters
    ----------
    vector : np.ndarray[tuple[int], np.dtype[np.complex_]]
        the vector to convert
    from_config : _BC0Inv
    to_config : _BC1Inv

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.complex_]]
    """
    util = BasisConfigUtil[Any, Any, Any](initial_basis)
    stacked = vector.reshape(util.shape)
    for i in range(3):
        matrix = get_basis_conversion_matrix(initial_basis[i], final_basis[i])
        # Each product gets moved to the end,
        # so the 0th index of stacked corresponds to the ith axis
        stacked = np.tensordot(stacked, matrix, axes=([0], [0]))

    return stacked.flatten()
