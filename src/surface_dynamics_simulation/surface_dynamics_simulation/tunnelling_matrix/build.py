from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from surface_dynamics_simulation.hopping_matrix.hopping_matrix import HoppingMatrix

    from .tunnelling_matrix import TunnellingMatrix


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


def build_from_hopping_matrix(
    matrix: HoppingMatrix[_L0Inv], shape: tuple[_L1Inv, _L2Inv]
) -> TunnellingMatrix[tuple[_L1Inv, _L2Inv, _L0Inv]]:
    """Given a list of hopping coefficients build the tunnelling matrix for a grid of the given shape."""
    n_states = matrix.shape[0]
    final_shape: tuple[_L1Inv, _L2Inv, _L0Inv] = (shape[0], shape[1], matrix.shape[0])  # type: ignore[assignment]

    out = np.zeros((*final_shape, *final_shape))
    for ix0, ix1, n in np.ndindex(final_shape):
        for dix0 in [-1, 0, 1]:
            for dix1 in [-1, 0, 1]:
                for m in range(n_states):
                    # Calculate the hopping rate out from state ix0, ix1, n
                    # in to the state jx0, jx1 m
                    jx0 = (ix0 + dix0) % out.shape[0]
                    jx1 = (ix1 + dix1) % out.shape[1]

                    hop_idx = np.ravel_multi_index((dix0, dix1), (3, 3), mode="wrap")
                    rate = matrix[n, m, hop_idx]
                    # Add the contribution from this rate to the total

                    # negative sign as this is a rate out from site ix0, ix1, n
                    # due to the occupation at ix0, ix1, n
                    out[ix0, ix1, n, ix0, ix1, n] -= rate
                    # positive sign as this is a rate in to site jx0, jx1, m
                    out[jx0, jx1, m, ix0, ix1, n] += rate

    vector = out.reshape(np.prod(final_shape), np.prod(final_shape))
    return {"shape": final_shape, "array": vector}
