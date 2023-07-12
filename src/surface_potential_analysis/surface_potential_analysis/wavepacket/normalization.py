from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from surface_potential_analysis.wavepacket.wavepacket import (
        Wavepacket,
    )

    _WInv = TypeVar("_WInv", bound=Wavepacket[Any, Any])


def calculate_normalization(wavepacket: _WInv) -> float:
    """
    calculate the normalization of a wavepacket.

    This should always be 1
    Parameters
    ----------
    wavepacket : Wavepacket[Any]

    Returns
    -------
    float
    """
    n_states = np.prod(wavepacket["eigenvalues"].shape)
    total_norm: complex = np.sum(np.conj(wavepacket["vectors"]) * wavepacket["vectors"])
    return float(total_norm / n_states)
